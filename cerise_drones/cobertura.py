"""Planejamento de cobertura por subarea (boustrophedon), transito evitando zonas proibidas,
estimativa de tempo de missao e checagem de bateria.

Convencao: o angulo de varredura `ang` (graus) e a direcao das linhas de voo medida a partir do
eixo x (leste) no sentido anti-horario. Internamente a regiao e girada de -ang para que as linhas
fiquem horizontais.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box

from .frota import Drone
from .geo import poligonos

EPS = 1e-9


# ----------------------------------------------------------------------------------------------
# Transito: menor caminho evitando obstaculos (grafo de visibilidade)
# ----------------------------------------------------------------------------------------------
class Navegador:
    """Calcula caminhos de transito que nao atravessam as zonas proibidas."""

    def __init__(self, obstaculos, contorno=None):
        self.obst = obstaculos if (obstaculos is not None and not obstaculos.is_empty) else None
        self.contorno = contorno  # zonas com margem: seus vertices sao os nos do grafo
        self._nos = None
        if self.obst is not None:
            shapely.prepare(self.obst)
            base = contorno if contorno is not None else self.obst
            nos = []
            for p in poligonos(base):
                ext = p.exterior.simplify(0.5)
                nos.extend(list(ext.coords)[:-1])
            self._nos = np.array(nos, dtype=float)
            self._vis = None

    def livre(self, p, q) -> bool:
        if self.obst is None:
            return True
        if math.dist(p, q) < 1e-6:
            return True
        return not self.obst.intersects(LineString([p, q]))

    def _visibilidade(self):
        if self._vis is None:
            n = len(self._nos)
            vis = [[] for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = tuple(self._nos[i]), tuple(self._nos[j])
                    if self.livre(a, b):
                        d = math.dist(a, b)
                        vis[i].append((j, d))
                        vis[j].append((i, d))
            self._vis = vis
        return self._vis

    def caminho(self, p, q) -> list[tuple]:
        """Lista de pontos de p ate q (inclusive) evitando obstaculos."""
        p, q = tuple(map(float, p)), tuple(map(float, q))
        if self.livre(p, q):
            return [p, q]
        vis = self._visibilidade()
        n = len(self._nos)
        nos = [tuple(x) for x in self._nos]
        ini, fim = n, n + 1
        viz_p = [(j, math.dist(p, nos[j])) for j in range(n) if self.livre(p, nos[j])]
        viz_q = {j: math.dist(q, nos[j]) for j in range(n) if self.livre(q, nos[j])}
        dist = {ini: 0.0}
        prev = {}
        heap = [(0.0, ini)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == fim:
                break
            if d > dist.get(u, math.inf):
                continue
            arestas = viz_p if u == ini else vis[u]
            for v, w in arestas:
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
            if u != ini and u in viz_q:
                nd = d + viz_q[u]
                if nd < dist.get(fim, math.inf):
                    dist[fim] = nd
                    prev[fim] = u
                    heapq.heappush(heap, (nd, fim))
        if fim not in prev:  # sem caminho (nao deveria ocorrer): segue em linha reta
            return [p, q]
        seq, u = [], fim
        while u != ini:
            seq.append(u)
            u = prev[u]
        seq.reverse()
        pts = [p] + [nos[k] for k in seq if k < n] + [q]
        return pts


# ----------------------------------------------------------------------------------------------
# Rota
# ----------------------------------------------------------------------------------------------
@dataclass
class Rota:
    drone_id: str
    pontos: np.ndarray                 # (K, 2) metros UTM, inclui a base no inicio e no fim
    cobre: np.ndarray                  # (K-1,) bool: perna k->k+1 e de cobertura (camera ativa)
    angulo: float = 0.0
    espacamento: float = 0.0
    regiao: object = None
    info: dict = field(default_factory=dict)

    @property
    def comprimentos(self) -> np.ndarray:
        return np.linalg.norm(np.diff(self.pontos, axis=0), axis=1)

    @property
    def distancia_m(self) -> float:
        return float(self.comprimentos.sum())

    @property
    def distancia_cobertura_m(self) -> float:
        return float(self.comprimentos[self.cobre].sum())

    def curvas_mask(self, limiar_graus: float = 20.0) -> np.ndarray:
        """Para cada ponto interno, se ha mudanca de rumo > limiar (pernas nulas ignoradas)."""
        P = self.pontos
        K = len(P)
        mask = np.zeros(K, dtype=bool)
        if K < 3:
            return mask
        v = np.diff(P, axis=0)
        L = np.linalg.norm(v, axis=1)
        idx = [i for i in range(len(v)) if L[i] > 1e-6]
        for a, b in zip(idx[:-1], idx[1:]):
            c = np.dot(v[a], v[b]) / (L[a] * L[b])
            if math.degrees(math.acos(max(-1.0, min(1.0, c)))) > limiar_graus:
                mask[b] = True  # curva no ponto inicial da perna b
        return mask

    @property
    def n_curvas(self) -> int:
        return int(self.curvas_mask().sum())

    def pernas_cobertura(self) -> list[tuple]:
        P = self.pontos
        return [(tuple(P[k]), tuple(P[k + 1])) for k in np.nonzero(self.cobre)[0]]


def tempo_basico(rota: Rota, drone: Drone, no_ar: bool = False) -> float:
    """Tempo sem trocas de bateria: subida + percurso + curvas + descida (s)."""
    t = rota.distancia_m / drone.velocidade_ms + rota.n_curvas * drone.tempo_curva_s
    t += drone.altitude_m / drone.vel_descida_ms
    if not no_ar:
        t += drone.altitude_m / drone.vel_subida_ms
    return t


# ----------------------------------------------------------------------------------------------
# Boustrophedon
# ----------------------------------------------------------------------------------------------
def _intervalos(geom):
    out = []
    for g in getattr(geom, "geoms", [geom]):
        if isinstance(g, LineString) and not g.is_empty and g.length > 0.5:
            xs = [c[0] for c in g.coords]
            out.append((min(xs), max(xs)))
    return out


def _linhas_varredura(reg_rot, s: float, comp: float = 0.0, obst_rot=None):
    """Linhas horizontais igualmente espacadas (<= s). Cada linha e responsavel por uma faixa de
    altura s2 centrada nela; o trecho voado cobre a extensao em x de cada pedaco da faixa (descontada
    meia pegada longitudinal `comp`/2), o que fecha as lacunas junto a bordas inclinadas sem voar alem
    do necessario."""
    minx, miny, maxx, maxy = reg_rot.bounds
    H = maxy - miny
    n = max(1, int(math.ceil(H / s - 1e-6)))
    s2 = H / n
    linhas = []
    for k in range(n):
        y = miny + s2 * (k + 0.5)
        faixa = reg_rot.intersection(box(minx - 1.0, y - s2 / 2, maxx + 1.0, y + s2 / 2))
        segs = []
        for peca in poligonos(faixa):
            if peca.area < 1.0:
                continue
            bx0, _, bx1, _ = peca.bounds
            # a pegada cobre comp/2 alem de cada extremidade: basta voar [bx0+comp/2, bx1-comp/2]
            x0, x1 = bx0 + comp / 2, bx1 - comp / 2
            if x0 > x1:
                x0 = x1 = (bx0 + bx1) / 2
            segs.append((x0, x1))
        segs.sort()
        fund = []
        for a, b in segs:  # funde intervalos sobrepostos
            if fund and a <= fund[-1][1] + 1e-6:
                fund[-1] = (fund[-1][0], max(fund[-1][1], b))
            else:
                fund.append((a, b))
        if obst_rot is not None and fund:
            # remove os trechos da linha que atravessam zonas proibidas (a perna e dividida)
            buracos = _intervalos(obst_rot.intersection(LineString([(minx - 1.0, y), (maxx + 1.0, y)])))
            for h0, h1 in buracos:
                novo = []
                for a, b in fund:
                    if b <= h0 or a >= h1:
                        novo.append((a, b))
                    else:
                        if a < h0 - 0.5:
                            novo.append((a, h0 - 0.5))
                        if b > h1 + 0.5:
                            novo.append((h1 + 0.5, b))
                fund = novo
        linhas.append((y, fund))
    return linhas, s2


def _celulas(linhas):
    """Agrupa segmentos de linhas consecutivas em celulas monotonas (decomposicao boustrophedon
    simplificada): um segmento continua a celula do segmento da linha anterior quando a
    sobreposicao em x e um-para-um."""
    celulas = []
    prev = []  # lista de (x0, x1, idx_celula)
    for y, segs in linhas:
        atual = []
        ov = [[(a0 <= b1 and b0 <= a1) for (b0, b1, _) in prev] for (a0, a1) in segs]
        for j, (a0, a1) in enumerate(segs):
            cand = [i for i, o in enumerate(ov[j]) if o]
            usado = False
            if len(cand) == 1:
                i = cand[0]
                n_ov_prev = sum(1 for jj in range(len(segs)) if ov[jj][i])
                if n_ov_prev == 1:
                    c = prev[i][2]
                    celulas[c].append((y, a0, a1))
                    atual.append((a0, a1, c))
                    usado = True
            if not usado:
                celulas.append([(y, a0, a1)])
                atual.append((a0, a1, len(celulas) - 1))
        prev = atual
    return celulas


def _percurso_celula(cel, inicio_baixo: bool, inicio_esq: bool):
    linhas = cel if inicio_baixo else cel[::-1]
    pts, cobre = [], []
    esq = inicio_esq
    for (y, x0, x1) in linhas:
        a, b = ((x0, y), (x1, y)) if esq else ((x1, y), (x0, y))
        if pts:
            cobre.append(False)  # ligacao entre linhas (curva)
        pts.append(a)
        pts.append(b)
        cobre.append(True)
        esq = not esq
    return pts, cobre


def _rot_pts(pts, ang):
    """Gira pontos (do sistema girado) de +ang de volta ao sistema UTM."""
    a = math.radians(ang)
    c, s = math.cos(a), math.sin(a)
    return [(x * c - y * s, x * s + y * c) for (x, y) in pts]


def _unrot_pt(p, ang):
    a = math.radians(-ang)
    c, s = math.cos(a), math.sin(a)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


def percurso_boustrophedon(regiao, ang: float, s: float, pos_ini, nav: Navegador | None = None,
                           comp: float = 0.0):
    """Gera os pontos de cobertura da regiao (sem a base): retorna (pontos, cobre, pos_final).

    Celulas sao visitadas de forma gulosa (entrada mais proxima da posicao atual) e as ligacoes
    entre celulas usam o Navegador para nao cruzar zonas proibidas."""
    reg_rot = affinity.rotate(regiao, -ang, origin=(0, 0))
    obst_rot = None
    if nav is not None and nav.obst is not None:
        bb = regiao.buffer(s).envelope
        if nav.obst.intersects(bb):
            obst_rot = affinity.rotate(nav.obst.intersection(bb), -ang, origin=(0, 0))
    linhas, s2 = _linhas_varredura(reg_rot, s, comp, obst_rot)
    cels = _celulas(linhas)
    pos_rot = _unrot_pt(pos_ini, ang)
    restantes = list(range(len(cels)))
    pts_rot, cobre = [], []
    atual = pos_rot
    while restantes:
        melhor = None
        for ci in restantes:
            cel = cels[ci]
            for baixo in (True, False):
                lin = cel[0] if baixo else cel[-1]
                for esq in (True, False):
                    e = (lin[1], lin[0]) if esq else (lin[2], lin[0])
                    d = math.dist(atual, e)
                    if melhor is None or d < melhor[0]:
                        melhor = (d, ci, baixo, esq)
        _, ci, baixo, esq = melhor
        p, c = _percurso_celula(cels[ci], baixo, esq)
        if pts_rot:
            cobre.append(False)  # transito entre celulas
        pts_rot.extend(p)
        cobre.extend(c)
        atual = p[-1]
        restantes.remove(ci)
    pts = _rot_pts(pts_rot, ang)
    # ligacoes (transito/curvas) que cruzam zonas proibidas sao desviadas
    if nav is not None and nav.obst is not None and pts:
        novo_p, novo_c = [pts[0]], []
        for k in range(len(pts) - 1):
            a, b = pts[k], pts[k + 1]
            if not cobre[k] and not nav.livre(a, b):
                cam = nav.caminho(a, b)
                for q in cam[1:]:
                    novo_p.append(q)
                    novo_c.append(False)
            else:
                novo_p.append(b)
                novo_c.append(cobre[k])
        pts, cobre = novo_p, novo_c
    return pts, cobre, s2


def montar_rota(drone: Drone, regiao, ang: float, s: float, nav: Navegador | None = None,
                inicio=None, fim=None) -> Rota:
    """Rota completa: inicio (base ou posicao atual) -> cobertura -> fim (base)."""
    inicio = tuple(inicio if inicio is not None else drone.partida_xy)
    fim = tuple(fim if fim is not None else drone.partida_xy)
    pts_cov, cob_cov, s2 = percurso_boustrophedon(regiao, ang, s, inicio, nav, drone.comprimento_pegada_m)
    if not pts_cov:
        P = np.array([inicio, fim], dtype=float)
        return Rota(drone.id, P, np.zeros(1, dtype=bool), ang, s, regiao, {"espacamento_real": s2})
    caminho_ida = nav.caminho(inicio, pts_cov[0]) if nav else [inicio, pts_cov[0]]
    caminho_volta = nav.caminho(pts_cov[-1], fim) if nav else [pts_cov[-1], fim]
    pts = list(caminho_ida[:-1]) + pts_cov + list(caminho_volta[1:])
    cobre = [False] * (len(caminho_ida) - 1) + cob_cov + [False] * (len(caminho_volta) - 1)
    P = np.array(pts, dtype=float)
    return Rota(drone.id, P, np.array(cobre, dtype=bool), ang, s, regiao, {"espacamento_real": s2})


def angulos_candidatos(regiao, passo_grade: float = 15.0, max_arestas: int = 8) -> list[float]:
    """Direcoes candidatas: arestas mais longas do fecho convexo (a varredura paralela a uma aresta
    do fecho minimiza a largura transversal), arestas do retangulo minimo e uma grade regular."""
    cand = list(np.arange(0.0, 180.0, passo_grade))
    hull = regiao.convex_hull
    if isinstance(hull, Polygon):
        c = np.array(hull.exterior.coords)
        ar = []
        for a, b in zip(c[:-1], c[1:]):
            L = math.dist(a, b)
            if L > 1e-6:
                ar.append((L, math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0))
        ar.sort(reverse=True)
        cand += [x for _, x in ar[:max_arestas]]
        mrr = regiao.minimum_rotated_rectangle
        if isinstance(mrr, Polygon):
            m = np.array(mrr.exterior.coords)
            for a, b in zip(m[:2], m[1:3]):
                cand.append(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0)
    out = []
    for a in sorted(x % 180.0 for x in cand):
        if all(min(abs(a - b), 180 - abs(a - b)) > 1.0 for b in out):
            out.append(a)
    return out


def planejar_regiao(drone: Drone, regiao, sobreposicao_lateral: float, nav: Navegador | None = None,
                    angulo: float | None = None, inicio=None, fim=None, no_ar: bool = False,
                    candidatos_extra=()) -> Rota:
    """Escolhe a orientacao que minimiza o tempo estimado (percurso + curvas) e devolve a rota."""
    s = drone.espacamento_m(sobreposicao_lateral)
    if regiao is None or regiao.is_empty:
        ini = tuple(inicio if inicio is not None else drone.partida_xy)
        fim_ = tuple(fim if fim is not None else drone.partida_xy)
        return Rota(drone.id, np.array([ini, fim_], float), np.zeros(1, bool), 0.0, s, regiao)
    angs = [angulo] if angulo is not None else angulos_candidatos(regiao) + list(candidatos_extra)
    melhor = None
    for a in angs:
        r = montar_rota(drone, regiao, a, s, nav, inicio, fim)
        t = tempo_basico(r, drone, no_ar)
        if melhor is None or t < melhor[0] - 1e-9:
            melhor = (t, r)
    rota = melhor[1]
    rota.info["angulos_testados"] = len(angs)
    return rota


def melhor_angulo_global(livre, s: float, nav=None) -> float:
    """Direcao de varredura que minimiza percurso+curvas para a area inteira (usada para alinhar a
    grade da particao DARP e as faixas)."""
    d = Drone("ref", velocidade_ms=10.0)
    c = livre.centroid
    d.partida_xy = (c.x, c.y)
    melhor = None
    for a in angulos_candidatos(livre):
        r = montar_rota(d, livre, a, s, None)
        t = r.distancia_m / 10.0 + r.n_curvas * d.tempo_curva_s
        if melhor is None or t < melhor[0]:
            melhor = (t, a)
    return melhor[1]


# ----------------------------------------------------------------------------------------------
# Bateria: simulacao ao longo da rota com retornos para troca
# ----------------------------------------------------------------------------------------------
@dataclass
class ResultadoMissao:
    tempo_total_s: float
    n_voos: int
    tempo_voo_s: float
    eventos: list           # (t, tipo, (x, y))
    viavel_um_voo: bool
    pontos_exec: np.ndarray  # trajetoria executada (inclui idas/voltas para troca)
    tempos_exec: np.ndarray  # instante de chegada em cada ponto de pontos_exec
    cobre_exec: np.ndarray   # perna k->k+1 de pontos_exec cobre?


def _densificar(P, cobre, passo):
    novo_p, novo_c = [P[0]], []
    for k in range(len(P) - 1):
        a, b = P[k], P[k + 1]
        L = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(L / passo)))
        for i in range(1, n + 1):
            novo_p.append(a + (b - a) * (i / n))
            novo_c.append(cobre[k])
    return np.array(novo_p), np.array(novo_c, dtype=bool)


def simular_missao(rota: Rota, drone: Drone, consumo_inicial_s: float = 0.0, no_ar: bool = False,
                   t0: float = 0.0, passo_m: float = 25.0) -> ResultadoMissao:
    """Percorre a rota com o modelo de energia por tempo de voo. Antes de cada trecho verifica se,
    apos executa-lo, ainda ha bateria util para voltar a base e pousar; se nao houver, volta, troca
    a bateria (tempo_troca_bateria_s), decola e retoma do ponto onde parou."""
    v = drone.velocidade_ms
    home = np.array(drone.partida_xy, dtype=float)
    util = drone.autonomia_util_s
    desc = drone.altitude_m / drone.vel_descida_ms
    sub = drone.altitude_m / drone.vel_subida_ms
    curvas = rota.curvas_mask()
    # mapeia curvas nos pontos originais para os densificados
    P, C = _densificar(rota.pontos, rota.cobre, passo_m)
    curva_d = np.zeros(len(P), dtype=bool)
    k_d = 0
    for k in range(len(rota.pontos) - 1):
        L = float(np.linalg.norm(rota.pontos[k + 1] - rota.pontos[k]))
        n = max(1, int(math.ceil(L / passo_m)))
        if curvas[k]:
            curva_d[k_d] = True
        k_d += n
    t = t0
    usado = consumo_inicial_s
    eventos = []
    exec_p, exec_t, exec_c = [P[0].copy()], [t], []
    voo = 0.0
    if not no_ar:
        t += sub
        usado += sub
        voo += sub
        eventos.append((t, "decolagem", tuple(P[0])))
        exec_t[0] = t
    n_voos = 1
    for k in range(len(P) - 1):
        a, b = P[k], P[k + 1]
        dt = float(np.linalg.norm(b - a)) / v + (drone.tempo_curva_s if curva_d[k] else 0.0)
        volta_b = float(np.linalg.norm(b - home)) / v + desc
        if usado + dt + volta_b > util + 1e-6 and k < len(P) - 2:
            # volta para trocar a bateria a partir de a
            ta = float(np.linalg.norm(a - home)) / v
            t += ta
            exec_p.append(home.copy()); exec_t.append(t); exec_c.append(False)
            t += desc
            voo += ta + desc
            eventos.append((t, "troca_bateria", tuple(a)))
            t += drone.tempo_troca_bateria_s + sub
            usado = sub
            voo += sub
            n_voos += 1
            tr = float(np.linalg.norm(a - home)) / v
            t += tr
            usado += tr
            voo += tr
            exec_p.append(a.copy()); exec_t.append(t); exec_c.append(False)
        t += dt
        usado += dt
        voo += dt
        exec_p.append(b.copy()); exec_t.append(t); exec_c.append(bool(C[k]))
    t += desc
    voo += desc
    exec_p.append(P[-1].copy()); exec_t.append(t); exec_c.append(False)  # pouso (descida no lugar)
    eventos.append((t, "pouso", tuple(P[-1])))
    tempo_total = t - t0
    viavel = (n_voos == 1) and (consumo_inicial_s + tempo_basico(rota, drone, no_ar) <= util + 1e-6)
    return ResultadoMissao(tempo_total, n_voos, voo, eventos, viavel, np.array(exec_p),
                           np.array(exec_t), np.array(exec_c, dtype=bool))
