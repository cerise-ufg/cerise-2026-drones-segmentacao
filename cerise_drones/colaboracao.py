"""Colaboracao entre drones: realocacao dinamica quando um drone falha / fica com bateria baixa.

Estrategias comparadas (evento: o drone `f` abandona a missao no instante t_f):
  'nenhuma'    : os demais seguem o plano original; a parte nao varrida da subarea de f e perdida.
  'reparticao' : apos a deteccao (timeout de HEARTBEAT), a area AINDA NAO COBERTA por todos os drones
                 e reparticionada (DARP ponderado + refinamento por tempo) entre os sobreviventes, a
                 partir das suas posicoes atuais; cada um recebe um AREA_ASSIGN (+ AREA_POLY).
  'leilao'     : a area orfa (restante de f) e dividida em tarefas (celulas de grade); leilao
                 sequencial de item unico (SSI) com objetivo minimax: a cada rodada cada sobrevivente
                 oferta (TASK_BID) o instante em que concluiria a tarefa apos sua fila atual e a menor
                 oferta vence (TASK_AWARD). As tarefas ganhas sao varridas apos a propria subarea.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Point, box

from . import protocolo as P
from .cobertura import planejar_regiao, simular_missao
from .geo import poligonos, so_poligonos
from .metricas import cobertura_raster, pegadas_pernas
from .particao import particionar
from .planejador import navegador_da_area


# ----------------------------------------------------------------------------------------------
@dataclass
class Trajetoria:
    """Trajetoria temporizada de um drone: pontos P (K,2), instantes T (K,), pernas de cobertura C."""
    P: np.ndarray
    T: np.ndarray
    C: np.ndarray

    @classmethod
    def da_missao(cls, m):
        return cls(m.pontos_exec.copy(), m.tempos_exec.copy(), m.cobre_exec.copy())

    @property
    def fim(self):
        return float(self.T[-1])

    def pos(self, t):
        if t <= self.T[0]:
            return tuple(self.P[0])
        if t >= self.T[-1]:
            return tuple(self.P[-1])
        k = int(np.searchsorted(self.T, t) - 1)
        dt = self.T[k + 1] - self.T[k]
        a = 0.0 if dt <= 0 else (t - self.T[k]) / dt
        return tuple(self.P[k] + a * (self.P[k + 1] - self.P[k]))

    def truncar(self, t):
        """Parte da trajetoria ate o instante t (inclusive o ponto interpolado em t)."""
        k = int(np.searchsorted(self.T, t, side="right"))
        P = list(self.P[:k]); T = list(self.T[:k]); C = list(self.C[:max(0, k - 1)])
        if k < len(self.T) and (not T or T[-1] < t):
            P.append(np.array(self.pos(t))); T.append(t)
            C.append(bool(self.C[k - 1]) if k >= 1 else False)
        return Trajetoria(np.array(P), np.array(T), np.array(C, dtype=bool))

    def concatenar(self, outra):
        return Trajetoria(np.vstack([self.P, outra.P[1:]]), np.concatenate([self.T, outra.T[1:]]),
                          np.concatenate([self.C, outra.C]))

    def pernas_ate(self, t):
        tr = self.truncar(t) if t < self.fim else self
        P, C = tr.P, tr.C
        return [(tuple(P[k]), tuple(P[k + 1])) for k in range(len(C)) if C[k]]

    def fim_cobertura(self):
        """Instante e indice do fim da ultima perna de cobertura."""
        idx = np.nonzero(self.C)[0]
        if len(idx) == 0:
            return float(self.T[0]), 0
        k = int(idx[-1]) + 1
        return float(self.T[k]), k


def _consumo(m, t, drone):
    """Tempo de voo consumido da bateria atual no instante t (considera trocas ja feitas)."""
    ini = 0.0
    for (te, tipo, _) in m.eventos:
        if tipo == "troca_bateria" and te <= t:
            ini = te + drone.tempo_troca_bateria_s
    return max(0.0, t - ini)


def _limpar(geom, s_ref):
    """Remove lascas: abertura morfologica de 1 m e descarte de partes < (s_ref/2)^2."""
    g = geom.buffer(-1.0).buffer(1.0)
    return so_poligonos(shapely.union_all([p for p in poligonos(g) if p.area >= 0.25 * s_ref * s_ref]))


def _pegada(pernas, d):
    return shapely.union_all(pegadas_pernas(pernas, d.largura_pegada_m, d.comprimento_pegada_m)) if pernas else None


class _AreaParcial:
    """Visao da AreaMissao restrita a uma sub-geometria (para reparticionar o nao coberto)."""

    def __init__(self, area, livre):
        self.__dict__.update({k: v for k, v in area.__dict__.items()})
        self.livre = livre



def _alocar_componentes(comps, frota_s, sobrep, k=3):
    """Distribui os drones sobreviventes entre as componentes desconexas da area nao coberta,
    minimizando (por enumeracao) o maior tempo estimado = deslocamento ate a componente + area da
    componente / soma das taxas de cobertura dos drones alocados a ela. Se houver mais componentes que
    drones, as menores sao anexadas ao drone de menor tempo estimado."""
    import itertools
    N, m = len(frota_s), len(comps)
    taxa = [d.taxa_cobertura_m2s(sobrep) for d in frota_s]
    areas = [c.area for c in comps]
    cent = [c.centroid for c in comps]
    ordem = sorted(range(m), key=lambda c: -areas[c])
    principais = ordem[:N]
    extras = ordem[N:]
    cands = []
    for atrib in itertools.product(principais, repeat=N):
        if len(set(atrib)) < len(principais):
            continue
        T = 0.0
        for c in principais:
            ids = [i for i in range(N) if atrib[i] == c]
            tc = areas[c] / sum(taxa[i] for i in ids)
            for i in ids:
                x, y = frota_s[i].partida_xy
                T = max(T, math.dist((x, y), (cent[c].x, cent[c].y)) / frota_s[i].velocidade_ms + tc)
        cands.append((T, atrib))
    cands.sort(key=lambda x: x[0])
    anexos = {i: [] for i in range(N)}
    for c in extras:
        i = min(range(N), key=lambda i: math.dist(frota_s[i].partida_xy, (cent[c].x, cent[c].y)))
        anexos[i].append(c)
    return [list(a) for _, a in cands[:k]], anexos


def _reparticionar(area, comps, atrib, anexos, frota_s, pesos, sobrep, ang):
    """Particao da area nao coberta (componentes `comps`) entre os sobreviventes, dada a alocacao
    drone -> componente (DARP dentro de cada componente com mais de um drone)."""
    if not comps:
        return [None] * len(frota_s)
    regioes = [[] for _ in frota_s]
    for c in set(atrib):
        ids = [i for i in range(len(frota_s)) if atrib[i] == c]
        if len(ids) == 1:
            regioes[ids[0]].append(comps[c])
            continue
        w = [pesos[i] for i in ids]
        sw = sum(w)
        part = particionar(_AreaParcial(area, comps[c]), [frota_s[i] for i in ids], "darp", sobrep,
                           sementes="bases", pesos=[x / sw for x in w], angulo_ref=ang)
        for i, g in zip(ids, part.regioes):
            regioes[i].append(g)
    for i, cs in anexos.items():
        regioes[i] += [comps[c] for c in cs]
    return [so_poligonos(shapely.union_all(r)) if r else None for r in regioes]

# ----------------------------------------------------------------------------------------------
@dataclass
class ResultadoFalha:
    estrategia: str
    falho: str
    t_falha_s: float
    makespan_base_s: float
    makespan_s: float
    aumento_makespan_pct: float
    area_perdida_m2: float
    cobertura_pct: float
    tempo_replanejamento_s: float
    mensagens: dict
    trajetorias: list
    regioes_novas: list = field(default_factory=list)
    detalhes: dict = field(default_factory=dict)


def simular_falha(plano, idx_falho: int, frac_falha: float = 0.5, estrategia: str = "reparticao",
                  t_deteccao_s: float = 3.0, sf: int = 9) -> ResultadoFalha:
    area, frota = plano.area, plano.frota
    trajs = [Trajetoria.da_missao(m) for m in plano.missoes]
    mk_base = max(t.fim for t in trajs)
    tf = frac_falha * trajs[idx_falho].fim
    tr = tf + t_deteccao_s
    df = frota[idx_falho]
    sobrev = [i for i in range(len(frota)) if i != idx_falho]
    s_ref = float(np.median([d.espacamento_m(plano.sobreposicao_lateral) for d in frota]))
    msgs = {"quadros": 0, "bytes": 0, "tempo_no_ar_ms": 0.0, "por_tipo": {}}

    def conta(qs):
        for q in qs:
            tipo = P.NOMES[q[0] & 0x0F]
            msgs["quadros"] += 1
            msgs["bytes"] += len(q)
            msgs["tempo_no_ar_ms"] += P.tempo_no_ar_ms(len(q), sf)
            msgs["por_tipo"][tipo] = msgs["por_tipo"].get(tipo, 0) + 1

    # o drone falho para em tf (pousa / RTL): sua trajetoria de cobertura termina ali
    traj_f = trajs[idx_falho].truncar(tf)
    novas = list(trajs)
    novas[idx_falho] = traj_f
    regioes_novas = []
    t0 = time.perf_counter()
    det = {}
    nav = navegador_da_area(area)
    if estrategia == "nenhuma":
        pass
    elif estrategia == "reparticao":
        # area ainda nao coberta por ninguem em tr (o falho so ate tf)
        cob = [_pegada(trajs[i].pernas_ate(tr), frota[i]) for i in sobrev] + [_pegada(traj_f.pernas_ate(tf), df)]
        cob = shapely.union_all([c for c in cob if c is not None])
        U = _limpar(area.livre.difference(cob), s_ref)
        det["area_nao_coberta_m2"] = U.area
        fr2 = []
        for i in sobrev:
            d = copy.copy(frota[i])
            d.partida_xy = trajs[i].pos(tr)
            fr2.append(d)
        from .frota import pesos_frota
        pesos = pesos_frota(fr2, "taxa", plano.sobreposicao_lateral)
        melhor = None
        comps = poligonos(U)
        alocs, anexos = _alocar_componentes(comps, fr2, plano.sobreposicao_lateral) if comps else ([[0] * len(fr2)], {})
        det["alocacoes_avaliadas"] = len(alocs)
        pesos0 = list(pesos)
        for atrib in alocs:
            pesos = list(pesos0)
            for it in range(3):  # refinamento pelo instante de conclusao (inclui bateria ja consumida)
                regs = _reparticionar(area, comps, atrib, anexos, fr2, pesos, plano.sobreposicao_lateral,
                                      plano.particao.angulo_ref)
                res = []
                for j, i in enumerate(sobrev):
                    d0 = frota[i]
                    rota = planejar_regiao(d0, regs[j], plano.sobreposicao_lateral, nav,
                                           inicio=fr2[j].partida_xy, fim=d0.partida_xy, no_ar=True,
                                           candidatos_extra=[plano.particao.angulo_ref])
                    m = simular_missao(rota, d0, consumo_inicial_s=_consumo(plano.missoes[i], tr, d0), no_ar=True, t0=tr)
                    res.append((rota, m))
                T = [tr + m.tempo_total_s for _, m in res]
                if melhor is None or max(T) < melhor[0]:
                    melhor = (max(T), regs, res)
                Tm = sum(T) / len(T)
                pesos = [x * ((Tm - tr) / max(1.0, t - tr)) ** 0.8 for x, t in zip(pesos, T)]
                sp = sum(pesos)
                pesos = [x / sp for x in pesos]
        _, regs, res = melhor
        regioes_novas = regs
        for j, i in enumerate(sobrev):
            rota, m = res[j]
            novas[i] = trajs[i].truncar(tr).concatenar(Trajetoria.da_missao(m))
            # mensagens: AREA_ASSIGN + AREA_POLY para cada sobrevivente
            if regs[j] is not None and not regs[j].is_empty:
                conta(P.area_assign_poligono(10 + j, 2, area.proj.lonlat(regs[j]), frota[i].altitude_m,
                                             frota[i].espacamento_m(plano.sobreposicao_lateral), src=P.GATEWAY,
                                             dst=int(frota[i].id[1:]), proj=area.proj))
    elif estrategia == "leilao":
        cob_f = _pegada(traj_f.pernas_ate(tf), df)
        orfa = plano.particao.regioes[idx_falho]
        O = _limpar(orfa.difference(cob_f) if cob_f is not None else orfa, s_ref)
        det["area_orfa_m2"] = O.area
        ang = plano.particao.angulo_ref
        cel = 2.0 * s_ref
        Or = affinity.rotate(O, -ang, origin=(0, 0))
        minx, miny, maxx, maxy = Or.bounds
        tarefas = []
        for yy in np.arange(miny, maxy, cel):
            for xx in np.arange(minx, maxx, cel):
                g = Or.intersection(box(xx, yy, xx + cel, yy + cel))
                if g.area > 0.05 * cel * cel:
                    tarefas.append(so_poligonos(affinity.rotate(g, ang, origin=(0, 0))))
        det["n_tarefas"] = len(tarefas)
        # anuncio: AREA_ASSIGN (broadcast) + AREA_POLY com o poligono da area orfa
        if not O.is_empty:
            conta(P.area_assign_poligono(99, 3, area.proj.lonlat(O), df.altitude_m, cel, src=P.GATEWAY,
                                         dst=P.BROADCAST, proj=area.proj))
        fila = {}
        for i in sobrev:
            t_fc, k = trajs[i].fim_cobertura()
            fila[i] = {"t": t_fc, "p": tuple(trajs[i].P[k]), "tarefas": []}
        restantes = list(range(len(tarefas)))
        rodadas = 0
        while restantes:
            rodadas += 1
            ofertas = []
            for i in sobrev:
                d = frota[i]
                taxa = d.taxa_cobertura_m2s(plano.sobreposicao_lateral)
                melhor_i = None
                for j in restantes:
                    c = tarefas[j].centroid
                    custo = fila[i]["t"] + math.dist(fila[i]["p"], (c.x, c.y)) / d.velocidade_ms + 1.25 * tarefas[j].area / taxa
                    if melhor_i is None or custo < melhor_i[0]:
                        melhor_i = (custo, j)
                ofertas.append((melhor_i[0], i, melhor_i[1]))
                conta([P.codificar(P.TASK_BID, (melhor_i[1] % 256, 99, min(65535, int(melhor_i[0] * 10) % 65536)),
                                   int(frota[i].id[1:]), P.GATEWAY)])
            custo, i, j = min(ofertas)
            conta([P.codificar(P.TASK_AWARD, (j % 256, 99, int(frota[i].id[1:])), P.GATEWAY, P.BROADCAST)])
            c = tarefas[j].centroid
            fila[i]["t"] = custo
            fila[i]["p"] = (c.x, c.y)
            fila[i]["tarefas"].append(j)
            restantes.remove(j)
        det["rodadas_leilao"] = rodadas
        regioes_novas = [None] * len(sobrev)
        for jj, i in enumerate(sobrev):
            if not fila[i]["tarefas"]:
                continue
            d = frota[i]
            reg = so_poligonos(shapely.union_all([tarefas[j] for j in fila[i]["tarefas"]]).buffer(0.01))
            regioes_novas[jj] = reg
            t_fc, k = trajs[i].fim_cobertura()
            rota = planejar_regiao(d, reg, plano.sobreposicao_lateral, nav, inicio=tuple(trajs[i].P[k]),
                                   fim=d.partida_xy, no_ar=True, candidatos_extra=[ang])
            m = simular_missao(rota, d, consumo_inicial_s=_consumo(plano.missoes[i], t_fc, d), no_ar=True, t0=t_fc)
            novas[i] = trajs[i].truncar(t_fc).concatenar(Trajetoria.da_missao(m))
    else:
        raise ValueError(estrategia)
    t_rep = time.perf_counter() - t0
    # deteccao da falha: HEARTBEATs perdidos (1 Hz) -> sem custo extra de mensagens
    mk = max(novas[i].fim for i in sobrev)
    peg = [_pegada(novas[i].pernas_ate(novas[i].fim), frota[i]) for i in range(len(frota))]
    cobr = cobertura_raster(area.livre, [g for g in peg if g is not None])
    perdida = area.livre.difference(shapely.union_all([g for g in peg if g is not None])).area
    return ResultadoFalha(estrategia, df.id, tf, mk_base, mk, 100.0 * (mk - mk_base) / mk_base, perdida,
                          cobr["cobertura_pct"], t_rep, msgs, novas, regioes_novas, det)
