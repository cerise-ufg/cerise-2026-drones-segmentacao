"""Metodos de particao (segmentacao) da area livre entre os drones da frota.

(a) faixas    : faixas paralelas a direcao de varredura, com larguras proporcionais aos pesos.
(b) kmeans    : k-means sobre pontos amostrados na area + celulas de Voronoi dos centroides
                recortadas pelo poligono (linha de base classica, nao ponderada).
(c) darp      : particao em grade no estilo DARP (Kapoutsis et al., 2017) ponderada pela
                capacidade de cada drone, com correcao de conectividade.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from shapely import affinity
from shapely.geometry import MultiPoint, Point, box

from .cobertura import melhor_angulo_global
from .frota import Drone, pesos_frota
from .geo import poligonos, so_poligonos


@dataclass
class ResultadoParticao:
    metodo: str
    regioes: list                 # geometria (metros) de cada drone, alinhada com a frota
    pesos: list
    tempo_s: float
    angulo_ref: float
    info: dict = field(default_factory=dict)

    def areas(self) -> list[float]:
        return [r.area for r in self.regioes]


# ----------------------------------------------------------------------------------------------
def _ordem_por_projecao(frota, ang):
    a = math.radians(ang)
    nrm = (-math.sin(a), math.cos(a))  # normal a direcao de varredura
    proj = [d.partida_xy[0] * nrm[0] + d.partida_xy[1] * nrm[1] for d in frota]
    return list(np.argsort(proj))


def particao_faixas(livre, frota, pesos, ang):
    """Faixas paralelas a direcao `ang`; as posicoes de corte sao achadas por bissecao de modo que a
    area de cada faixa seja proporcional ao peso do drone. Drones ordenados pela projecao da base
    na normal (drone mais 'abaixo' fica com a faixa mais 'abaixo')."""
    R = affinity.rotate(livre, -ang, origin=(0, 0))
    minx, miny, maxx, maxy = R.bounds
    A = R.area
    ordem = _ordem_por_projecao(frota, ang)
    cortes = [miny]
    acum = 0.0
    for idx in ordem[:-1]:
        acum += pesos[idx]
        lo, hi = miny, maxy
        alvo = acum * A
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            a = R.intersection(box(minx - 1, miny - 1, maxx + 1, mid)).area
            if a < alvo:
                lo = mid
            else:
                hi = mid
        cortes.append(0.5 * (lo + hi))
    cortes.append(maxy)
    regioes = [None] * len(frota)
    for k, idx in enumerate(ordem):
        faixa = R.intersection(box(minx - 1, cortes[k], maxx + 1, cortes[k + 1]))
        regioes[idx] = so_poligonos(affinity.rotate(faixa, ang, origin=(0, 0)))
    return regioes


def particao_kmeans(livre, frota, n_amostras=4000, seed=0):
    from sklearn.cluster import KMeans

    N = len(frota)
    passo = math.sqrt(livre.area / n_amostras)
    minx, miny, maxx, maxy = livre.bounds
    xs = np.arange(minx + passo / 2, maxx, passo)
    ys = np.arange(miny + passo / 2, maxy, passo)
    X, Y = np.meshgrid(xs, ys)
    X, Y = X.ravel(), Y.ravel()
    dentro = shapely.contains_xy(livre, X, Y)
    pts = np.column_stack([X[dentro], Y[dentro]])
    km = KMeans(n_clusters=N, n_init=10, random_state=seed).fit(pts)
    centros = km.cluster_centers_
    ext = livre.envelope.buffer(livre.length)
    cel = shapely.voronoi_polygons(MultiPoint([tuple(c) for c in centros]), extend_to=ext, ordered=True)
    cels = list(cel.geoms)
    regs = [so_poligonos(c.intersection(livre)) for c in cels]
    # atribuicao drone -> regiao (Hungaro, distancia base -> centroide)
    custo = np.array([[math.dist(d.partida_xy, tuple(centros[j])) for j in range(N)] for d in frota])
    li, cj = linear_sum_assignment(custo)
    regioes = [None] * N
    for i, j in zip(li, cj):
        regioes[i] = regs[j]
    return regioes, {"centros": centros.tolist()}


# ----------------------------------------------------------------------------------------------
def _grade(livre, ang, cel_x, cel_y):
    """Grade de celulas retangulares alinhada a direcao `ang` (cel_x ao longo da varredura, cel_y
    transversal); devolve as celulas validas (que tocam a area livre), a fracao de area livre de cada
    uma e a geometria recortada (no sistema girado). As arestas sao calculadas uma unica vez para que
    celulas vizinhas compartilhem coordenadas identicas (uniao sem micro-frestas)."""
    R = affinity.rotate(livre, -ang, origin=(0, 0))
    minx, miny, maxx, maxy = R.bounds
    nx = int(math.ceil((maxx - minx) / cel_x))
    ny = int(math.ceil((maxy - miny) / cel_y))
    xs = minx + np.arange(nx + 1) * cel_x
    ys = miny + np.arange(ny + 1) * cel_y
    ii, jj = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    caixas = shapely.box(xs[jj], ys[ii], xs[jj + 1], ys[ii + 1])
    shapely.prepare(R)
    inter = shapely.intersection(caixas, R)
    areas = shapely.area(inter)
    valid = areas > 1e-3 * cel_x * cel_y
    return dict(R=R, nx=nx, ny=ny, minx=minx, miny=miny, cel_x=cel_x, cel_y=cel_y, valid=valid,
                ii=ii[valid], jj=jj[valid], frac=areas[valid] / (cel_x * cel_y), geoms=inter[valid])


def _grafo(g, diagonais=True):
    """Grafo de vizinhanca (8 ou 4 vizinhos) entre celulas validas (pesos = distancias em metros)."""
    ny, nx = g["ny"], g["nx"]
    idx = -np.ones((ny, nx), dtype=int)
    idx[g["ii"], g["jj"]] = np.arange(len(g["ii"]))
    rows, cols, w = [], [], []
    cx, cy = g["cel_x"], g["cel_y"]
    dg = math.hypot(cx, cy)
    viz = [(0, 1, cx), (1, 0, cy)] + ([(1, 1, dg), (1, -1, dg)] if diagonais else [])
    for di, dj, peso in viz:
        a = idx[0:ny - di, max(0, -dj):nx - max(0, dj)]
        b = idx[di:ny, max(0, dj):nx - max(0, -dj)]
        m = (a >= 0) & (b >= 0)
        rows.append(a[m]); cols.append(b[m]); w.append(np.full(m.sum(), peso))
    rows = np.concatenate(rows); cols = np.concatenate(cols); w = np.concatenate(w)
    M = len(g["ii"])
    G = coo_matrix((np.concatenate([w, w]), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))), shape=(M, M)).tocsr()
    return G, idx


def _sementes(g, frota, ang, livre, seed=0, modo="bases"):
    """Celula inicial de cada drone.
    'bases'  : celula valida mais proxima da base do drone (DARP classico; a regiao 'cresce' a partir
               da base). Se duas bases caem na mesma vizinhanca (base comum), cai para 'centros'.
    'centros': centros de um k-means (ponderado pela area) da area livre, atribuidos aos drones pelo
               metodo Hungaro (distancia base -> centro); regioes mais compactas."""
    a = math.radians(-ang)
    c, s = math.cos(a), math.sin(a)
    cx = g["minx"] + (g["jj"] + 0.5) * g["cel_x"]
    cy = g["miny"] + (g["ii"] + 0.5) * g["cel_y"]
    bases = []
    for d in frota:
        x, y = d.partida_xy
        bases.append((x * c - y * s, x * s + y * c))
    sem = [int(np.argmin((cx - bx) ** 2 + (cy - by) ** 2)) for (bx, by) in bases]
    pos = np.column_stack([cx[sem], cy[sem]])
    dmin = min((math.dist(pos[i], pos[j]) for i in range(len(sem)) for j in range(i + 1, len(sem))), default=1e9)
    info = {"sementes": modo}
    if modo == "centros" or dmin < 2.5 * max(g["cel_x"], g["cel_y"]) or len(set(sem)) < len(sem):
        from sklearn.cluster import KMeans
        P = np.column_stack([cx, cy])
        km = KMeans(n_clusters=len(frota), n_init=10, random_state=seed).fit(P, sample_weight=g["frac"])
        cent = km.cluster_centers_
        custo = np.array([[math.dist(b, tuple(cc)) for cc in cent] for b in bases])
        li, cj = linear_sum_assignment(custo)
        sem = [0] * len(frota)
        for i, j in zip(li, cj):
            sem[i] = int(np.argmin((cx - cent[j][0]) ** 2 + (cy - cent[j][1]) ** 2))
        if modo != "centros":
            info["sementes"] = "centros (bases coincidentes)"
    return sem, info


def particao_darp(livre, frota, pesos, ang, cel_x, cel_y, max_iter=1500, tol=0.01, cc=0.01, eta=0.08,
                  ruido=1e-4, seed=0, sementes="bases", pesagem="aditiva"):
    """Particao em grade estilo DARP.

    pesagem='multiplicativa': E_i = m_i * C_i * D_i (formulacao original de Kapoutsis et al.; a
        conectividade e imposta pela matriz de correcao C_i e pode nao convergir em areas concavas com
        frotas heterogeneas).
    pesagem='aditiva': E_i = D_i - b_i (D_i = distancia geodesica na grade a partir da semente). As
        regioes resultantes sao estrelados geodesicos em torno da semente e portanto conexas por
        construcao; o balanceamento ajusta so os deslocamentos b_i (convergencia monotona).
    """
    rng = np.random.default_rng(seed)
    g = _grade(livre, ang, cel_x, cel_y)
    # grafo de 8 vizinhos (distancias geodesicas ~euclidianas); a conectividade das regioes e
    # verificada na conectividade-4 e ligacoes so diagonais sao resolvidas pela limpeza final
    G, idx = _grafo(g, diagonais=True)
    M = len(g["ii"])
    N = len(frota)
    sem, info = _sementes(g, frota, ang, livre, seed, sementes)
    amostra = (cel_y, cel_x)  # (linhas, colunas) para a transformada de distancia em metros
    D = dijkstra(G, indices=sem)
    if not np.all(np.isfinite(D)):  # area livre desconexa: completa com distancia euclidiana x 3
        cx = (g["jj"] + 0.5) * cel_x; cy = (g["ii"] + 0.5) * cel_y
        for i in range(N):
            ex = 3.0 * np.hypot(cx - cx[sem[i]], cy - cy[sem[i]])
            bad = ~np.isfinite(D[i])
            D[i, bad] = ex[bad] + np.nanmax(D[i][np.isfinite(D[i])])
    D = D + 0.5 * min(cel_x, cel_y)
    frac = g["frac"]
    alvo = np.asarray(pesos) * frac.sum()
    tol = max(tol, 1.5 / alvo.min())  # nao exige precisao menor que ~1,5 celula
    m = np.ones(N)
    b = np.zeros(N)
    escala = math.sqrt(livre.area / N)
    C = np.ones((N, M))
    img = -np.ones((g["ny"], g["nx"]), dtype=int)
    melhor = None
    melhor_qq = None  # melhor balanceamento mesmo que desconexo (corrigido depois pela limpeza)
    estrutura = ndimage.generate_binary_structure(2, 1)  # conectividade-4 (limpeza final)
    estrutura8 = estrutura
    it = 0
    hist = []
    for it in range(max_iter):
        if pesagem == "aditiva":
            E = D - b[:, None]
            if ruido > 0:
                E = E + ruido * escala * rng.standard_normal(E.shape)
        else:
            E = m[:, None] * C * D
            if ruido > 0:
                E = E * (1.0 + ruido * rng.standard_normal(E.shape))
        A = np.argmin(E, axis=0)
        A[sem] = np.arange(N)
        areas = np.bincount(A, weights=frac, minlength=N)
        img[g["ii"], g["jj"]] = A
        conectado = True
        for i in range(N):
            mask = img == i
            lab, nlab = ndimage.label(mask, structure=estrutura8)
            if nlab > 1:
                conectado = False
                ls = lab[g["ii"][sem[i]], g["jj"][sem[i]]]
                mR = lab == ls
                mQ = mask & ~mR
                dR = ndimage.distance_transform_edt(~mR, sampling=amostra)[g["ii"], g["jj"]]
                dQ = ndimage.distance_transform_edt(~mQ, sampling=amostra)[g["ii"], g["jj"]]
                v = dR - dQ
                vmin, vmax = v.min(), v.max()
                C[i] = (v - vmin) * (2 * cc / max(vmax - vmin, 1e-9)) + (1 - cc)
            else:
                C[i] = 1.0
        err = (areas - alvo) / alvo
        desb = float(np.max(np.abs(err)))
        hist.append(desb)
        chave = (0 if conectado else 1, desb)
        if melhor is None or chave < melhor[0]:
            melhor = (chave, A.copy(), it)
        if melhor_qq is None or desb < melhor_qq[0]:
            melhor_qq = (desb, A.copy())
        if conectado and desb < tol:
            break
        passo = eta / (1.0 + it / 300.0)  # passo decrescente evita oscilacao entre celulas vizinhas
        if pesagem == "aditiva":
            b = b - passo * escala * np.clip(err, -0.5, 0.5)
            b = b - b.mean()
        else:
            m = m * (1.0 + passo * np.clip(err, -0.5, 0.5))
            m = m / m.mean()
    def limpar(Aa):
        """componentes desconexas sao entregues ao vizinho com maior fronteira"""
        im = -np.ones((g["ny"], g["nx"]), dtype=int)
        im[g["ii"], g["jj"]] = Aa
        for _ in range(5):
            mudou = False
            for i in range(N):
                lab, nlab = ndimage.label(im == i, structure=estrutura)
                if nlab <= 1:
                    continue
                ls = lab[g["ii"][sem[i]], g["jj"][sem[i]]]
                for L in range(1, nlab + 1):
                    if L == ls:
                        continue
                    comp = lab == L
                    borda = ndimage.binary_dilation(comp, structure=estrutura) & ~comp & (im >= 0)
                    viz = im[borda]
                    viz = viz[viz != i]
                    if len(viz):
                        im[comp] = np.bincount(viz).argmax()
                        mudou = True
            if not mudou:
                break
        return im[g["ii"], g["jj"]]

    cands = [limpar(melhor[1])]
    if melhor_qq is not None and not (melhor[0][0] == 0 and melhor[0][1] < tol):
        cands.append(limpar(melhor_qq[1]))
    desbs = [float(np.max(np.abs((np.bincount(c, weights=frac, minlength=N) - alvo) / alvo))) for c in cands]
    A = cands[int(np.argmin(desbs))]
    img[g["ii"], g["jj"]] = A
    A = img[g["ii"], g["jj"]]
    regioes = []
    for i in range(N):
        geo = shapely.union_all(g["geoms"][A == i])
        geo = geo.buffer(1e-3, join_style="mitre").buffer(-1e-3, join_style="mitre")  # sela micro-frestas
        regioes.append(so_poligonos(affinity.rotate(geo, ang, origin=(0, 0))))
    areas = np.bincount(A, weights=frac, minlength=N)
    info.update({"iteracoes": it + 1, "celulas": int(M), "tamanho_celula_m": [cel_x, cel_y],
                 "desbalanceamento_final": float(np.max(np.abs((areas - alvo) / alvo))),
                 "conectado_sem_limpeza": bool(melhor[0][0] == 0), "pesagem": pesagem,
                 "historico_desb": hist[:: max(1, len(hist) // 200)]})
    return regioes, info


# ----------------------------------------------------------------------------------------------
def suavizar_fronteiras(livre, regioes, tol):
    """Retifica as fronteiras internas em 'escada' da particao em grade sem perder a propriedade de
    particao: as arestas internas (uniao das fronteiras menos o contorno da area) sao simplificadas
    (Douglas-Peucker, tolerancia `tol`), re-nodadas com o contorno original e poligonizadas; cada face
    resultante e atribuida a regiao original com a qual tem maior interseccao."""
    from shapely.ops import polygonize, unary_union, linemerge
    bordas = unary_union([r.boundary for r in regioes if r is not None and not r.is_empty])
    contorno = livre.boundary
    internas = bordas.difference(contorno.buffer(1e-6))
    try:
        internas = linemerge(internas) if internas.geom_type == "MultiLineString" else internas
    except Exception:
        pass
    simp = internas.simplify(tol, preserve_topology=True)
    rede = unary_union([simp, contorno])
    faces = [f for f in polygonize(rede) if f.area > 1e-3]
    novas = [[] for _ in regioes]
    shapely.prepare(livre)
    for f in faces:
        f = f.intersection(livre)
        if f.is_empty or f.area < 1e-3:
            continue
        inter = [f.intersection(r).area if r is not None else 0.0 for r in regioes]
        novas[int(np.argmax(inter))].append(f)
    out = [so_poligonos(shapely.union_all(n)) if n else r for n, r in zip(novas, regioes)]
    # protege contra falhas numericas: so aceita se a area total for preservada
    if abs(sum(o.area for o in out) - livre.area) > 1e-3 * livre.area:
        return regioes
    return out


def limpar_fragmentos(regioes, area_min: float):
    """Fragmentos pequenos (area < area_min) de uma regiao sao entregues a regiao vizinha com a qual
    compartilham mais fronteira (ou a mais proxima). Evita 'ilhas' que custam transito e curvas."""
    regioes = list(regioes)
    for _ in range(3):
        mudou = False
        for i, g in enumerate(regioes):
            partes = poligonos(g)
            if len(partes) <= 1:
                continue
            partes.sort(key=lambda p: -p.area)
            manter = [partes[0]] + [p for p in partes[1:] if p.area >= area_min]
            for frag in partes[1:]:
                if frag.area >= area_min:
                    continue
                melhor, mval = None, -1.0
                for j, h in enumerate(regioes):
                    if j == i or h is None or h.is_empty:
                        continue
                    L = frag.boundary.intersection(h.buffer(1e-6)).length
                    if L > mval:
                        melhor, mval = j, L
                if mval <= 1e-9:
                    melhor = min((j for j in range(len(regioes)) if j != i), key=lambda j: frag.distance(regioes[j]))
                regioes[melhor] = so_poligonos(shapely.union_all([regioes[melhor], frag.buffer(1e-6)]))
                mudou = True
            regioes[i] = so_poligonos(shapely.union_all(manter))
        if not mudou:
            break
    return regioes


def particionar(area, frota: list[Drone], metodo: str, sobreposicao_lateral: float = 0.2,
                modo_pesos: str = "taxa", angulo_ref: float | None = None, seed: int = 0,
                pesos: list | None = None, limpar: bool = True, **kw) -> ResultadoParticao:
    livre = area.livre
    if pesos is None:
        pesos = pesos_frota(frota, modo_pesos, sobreposicao_lateral)
    s_min = min(d.espacamento_m(sobreposicao_lateral) for d in frota)
    s_med = float(np.median([d.espacamento_m(sobreposicao_lateral) for d in frota]))
    t0 = time.perf_counter()
    if angulo_ref is None:
        angulo_ref = melhor_angulo_global(livre, s_med)
    info = {}
    if metodo == "faixas":
        regioes = particao_faixas(livre, frota, pesos, angulo_ref)
    elif metodo == "kmeans":
        regioes, info = particao_kmeans(livre, frota, seed=seed)
    elif metodo == "darp":
        # celulas quadradas de 1/3 do menor espacamento entre linhas, alinhadas a direcao de varredura
        cel = kw.pop("celula_m", None) or s_min / 3.0
        while livre.area / (cel * cel) > 6000:  # limita o custo em areas grandes
            cel *= 1.25
        suavizar = kw.pop("suavizar", True)
        regioes, info = particao_darp(livre, frota, pesos, angulo_ref, cel, cel, seed=seed, **kw)
        if suavizar:
            regioes = suavizar_fronteiras(livre, regioes, tol=1.5 * cel)
    else:
        raise ValueError(metodo)
    if limpar:
        regioes = limpar_fragmentos(regioes, 0.25 * s_min * s_min)
    dt = time.perf_counter() - t0
    return ResultadoParticao(metodo, regioes, pesos, dt, angulo_ref, info)
