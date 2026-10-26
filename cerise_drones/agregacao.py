"""Agregacao das contagens de pessoas entre drones.

1. Georreferenciamento (camera em nadir, terreno plano): cada deteccao (u, v) em pixels vira um ponto
   (x, y) UTM a partir da posicao/altitude/rumo do drone no instante da foto.
2. Posse: a deteccao so e contada pelo drone dono da subarea onde o ponto cai (as subareas sao
   disjuntas, logo nao ha dupla contagem entre drones fora da faixa de incerteza).
3. Deduplicacao: fotos consecutivas se sobrepoem (sobreposicao frontal/lateral), entao a mesma pessoa
   aparece varias vezes; deteccoes a menos de `raio_m` sao agrupadas (ligacao simples) e contadas uma
   vez. Na faixa de largura `banda_m` em torno das fronteiras entre subareas o agrupamento e feito
   entre drones diferentes (via COUNT_REPORT com as posicoes das deteccoes de borda), cobrindo o erro
   de georreferenciamento.
"""
from __future__ import annotations

import math

import numpy as np
import shapely
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def pixel_para_solo(u, v, largura_px, altura_px, pos_xy, altitude_m, hfov_graus, vfov_graus, rumo_graus):
    """Pixel -> solo para camera em nadir com o topo da imagem apontando para o rumo do voo
    (rumo medido a partir do norte, sentido horario)."""
    u = np.asarray(u, float); v = np.asarray(v, float)
    dx = (u - largura_px / 2) / (largura_px / 2) * altitude_m * math.tan(math.radians(hfov_graus) / 2)   # direita
    dy = (altura_px / 2 - v) / (altura_px / 2) * altitude_m * math.tan(math.radians(vfov_graus) / 2)     # frente
    r = math.radians(rumo_graus)
    leste = dx * math.cos(r) + dy * math.sin(r)
    norte = -dx * math.sin(r) + dy * math.cos(r)
    return pos_xy[0] + leste, pos_xy[1] + norte


def _agrupar(xy, raio):
    if len(xy) == 0:
        return np.zeros(0, dtype=int), 0
    t = cKDTree(xy)
    pares = t.query_pairs(raio, output_type="ndarray")
    from scipy.sparse import coo_matrix
    n = len(xy)
    A = coo_matrix((np.ones(len(pares)), (pares[:, 0], pares[:, 1])), shape=(n, n)) if len(pares) else coo_matrix((n, n))
    k, lab = connected_components(A, directed=False)
    return lab, k


def contar(deteccoes: np.ndarray, drone_det: np.ndarray, regioes: list, raio_m: float = 1.5, banda_m: float = 3.0) -> dict:
    """deteccoes: (n,2) UTM; drone_det: (n,) indice do drone que fez a deteccao.

    Retorna contagens por estrategia: soma ingenua, deduplicacao por drone sem posse, e o metodo
    proposto (posse + deduplicacao + agrupamento entre drones na faixa de fronteira)."""
    deteccoes = np.asarray(deteccoes, float).reshape(-1, 2)
    drone_det = np.asarray(drone_det, int)
    out = {"soma_ingenua": int(len(deteccoes))}
    # (b) cada drone deduplica as suas deteccoes, sem regra de posse
    tot = 0
    for i in range(len(regioes)):
        _, k = _agrupar(deteccoes[drone_det == i], raio_m)
        tot += k
    out["dedup_por_drone_sem_posse"] = int(tot)
    # (c) posse + dedup + borda
    X, Y = deteccoes[:, 0], deteccoes[:, 1]
    dono = np.full(len(deteccoes), -1)
    for i, g in enumerate(regioes):
        if g is None or g.is_empty:
            continue
        shapely.prepare(g)
        dono[shapely.contains_xy(g, X, Y)] = i
    fronteiras = shapely.union_all([g.boundary for g in regioes if g is not None and not g.is_empty])
    banda = fronteiras.buffer(banda_m)
    shapely.prepare(banda)
    na_banda = shapely.contains_xy(banda, X, Y)
    interior_ok = (dono == drone_det) & ~na_banda
    total = 0
    por_drone = []
    for i in range(len(regioes)):
        _, k = _agrupar(deteccoes[interior_ok & (drone_det == i)], raio_m)
        por_drone.append(int(k))
        total += k
    _, kb = _agrupar(deteccoes[na_banda], raio_m)  # entre drones, na faixa de fronteira
    out["proposto_posse_dedup_borda"] = int(total + kb)
    out["proposto_por_drone_interior"] = por_drone
    out["proposto_borda"] = int(kb)
    out["frac_deteccoes_na_borda"] = float(na_banda.mean()) if len(na_banda) else 0.0
    return out
