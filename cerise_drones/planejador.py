"""Orquestracao: particao + cobertura por subarea + bateria + metricas (+ exportacao)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .cobertura import Navegador, ResultadoMissao, Rota, planejar_regiao, simular_missao
from .frota import Drone
from .geo import AreaMissao
from .metricas import resumo_plano
from .particao import ResultadoParticao, particionar


@dataclass
class PlanoMissao:
    area: AreaMissao
    frota: list
    particao: ResultadoParticao
    rotas: list
    missoes: list
    metricas: dict
    tempo_planejamento_s: float
    sobreposicao_lateral: float = 0.2
    sobreposicao_frontal: float = 0.7

    def exportar(self, pasta, nome="missao"):
        from .exportar import exportar_tudo
        return exportar_tudo(pasta, self.area, self.frota, self.rotas, self.particao.regioes,
                             self.sobreposicao_frontal, nome)


def navegador_da_area(area: AreaMissao) -> Navegador | None:
    if area.zonas_buf.is_empty:
        return None
    return Navegador(area.obstaculos, contorno=area.zonas_buf)


def _planejar_rotas(area, frota, part, sobreposicao_lateral, nav):
    rotas, missoes = [], []
    for d, reg in zip(frota, part.regioes):
        r = planejar_regiao(d, reg, sobreposicao_lateral, nav, candidatos_extra=[part.angulo_ref])
        rotas.append(r)
        missoes.append(simular_missao(r, d))
    return rotas, missoes


def planejar_missao(area: AreaMissao, frota: list[Drone], metodo: str = "darp",
                    sobreposicao_lateral: float = 0.2, sobreposicao_frontal: float = 0.7,
                    modo_pesos: str = "taxa", seed: int = 0, refinar: int = 0, alfa: float = 0.8,
                    **kw) -> PlanoMissao:
    """Planeja a missao. Com `refinar` > 0, os pesos da particao sao reajustados iterativamente pelo
    tempo de missao estimado de cada drone (inclui transito da base, curvas, subida/descida e trocas
    de bateria): w_i <- w_i * (T_medio / T_i)^alfa, mantendo a melhor solucao (menor makespan)."""
    nav = navegador_da_area(area)
    t_part = t_plan = 0.0
    part = particionar(area, frota, metodo, sobreposicao_lateral, modo_pesos, seed=seed, **kw)
    t_part += part.tempo_s
    t0 = time.perf_counter()
    rotas, missoes = _planejar_rotas(area, frota, part, sobreposicao_lateral, nav)
    t_plan += time.perf_counter() - t0
    melhor = (max(m.tempo_total_s for m in missoes), part, rotas, missoes)
    pesos = list(part.pesos)
    hist = [melhor[0]]
    for _ in range(refinar):
        T = [m.tempo_total_s for m in missoes]
        Tm = sum(T) / len(T)
        pesos = [w * (Tm / t) ** alfa for w, t in zip(pesos, T)]
        sw = sum(pesos)
        pesos = [w / sw for w in pesos]
        part = particionar(area, frota, metodo, sobreposicao_lateral, modo_pesos, seed=seed,
                           angulo_ref=melhor[1].angulo_ref, pesos=pesos, **kw)
        t_part += part.tempo_s
        t0 = time.perf_counter()
        rotas, missoes = _planejar_rotas(area, frota, part, sobreposicao_lateral, nav)
        t_plan += time.perf_counter() - t0
        mk = max(m.tempo_total_s for m in missoes)
        hist.append(mk)
        if mk < melhor[0]:
            melhor = (mk, part, rotas, missoes)
    _, part, rotas, missoes = melhor
    met = resumo_plano(area, frota, rotas, missoes, part.regioes, part.pesos)
    nome_metodo = metodo + ("+ref" if refinar else "")
    met.update({"metodo": nome_metodo, "modo_pesos": modo_pesos, "n_drones": len(frota),
                "tempo_particao_s": t_part, "tempo_cobertura_s": t_plan,
                "tempo_algoritmo_s": t_part + t_plan, "angulo_ref_graus": part.angulo_ref,
                "area_livre_m2": area.livre.area, "historico_makespan_s": hist})
    if metodo == "darp":
        met["darp"] = {k: v for k, v in part.info.items() if k != "historico_desb"}
    return PlanoMissao(area, frota, part, rotas, missoes, met, t_plan, sobreposicao_lateral, sobreposicao_frontal)


def carregar_cenario_json(caminho) -> tuple[AreaMissao, list[Drone], dict]:
    """Le um arquivo de cenario (ver exemplos/cenario_ufg.json)."""
    caminho = Path(caminho)
    cfg = json.loads(caminho.read_text(encoding="utf-8"))
    base = caminho.parent

    def resolve(p):
        return p if isinstance(p, dict) else str((base / p).resolve())

    area = AreaMissao.de_arquivos(cfg.get("nome", caminho.stem), resolve(cfg["area"]),
                                  [resolve(z) for z in cfg.get("zonas_proibidas", [])],
                                  cfg.get("margem_zona_m", 10.0))
    frota = []
    for i, dd in enumerate(cfg["frota"]):
        dd = dict(dd)
        lon, lat = dd.pop("partida_lonlat")
        dd.setdefault("id", f"D{i + 1}")
        d = Drone(**dd)
        d.partida_xy = area.proj.ponto_xy(lon, lat)
        frota.append(d)
    return area, frota, cfg
