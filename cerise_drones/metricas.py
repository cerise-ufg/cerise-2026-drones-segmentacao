"""Metricas do plano: balanceamento, makespan, distancia, curvas e cobertura rasterizada."""
from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import Polygon

from .cobertura import ResultadoMissao, Rota
from .frota import Drone


def pegadas_pernas(pernas, largura: float, comprimento: float):
    """Retangulos varridos pela camera ao longo de cada perna de cobertura (largura transversal,
    estendidos de meia pegada longitudinal nas extremidades)."""
    polys = []
    for (a, b) in pernas:
        a = np.asarray(a, float); b = np.asarray(b, float)
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            u = np.array([1.0, 0.0])
        else:
            u = d / L
        n = np.array([-u[1], u[0]])
        a2 = a - u * comprimento / 2
        b2 = b + u * comprimento / 2
        w = largura / 2
        polys.append(Polygon([tuple(a2 + n * w), tuple(b2 + n * w), tuple(b2 - n * w), tuple(a2 - n * w)]))
    return polys


def pegada_rota(rota: Rota, drone: Drone):
    return shapely.union_all(pegadas_pernas(rota.pernas_cobertura(), drone.largura_pegada_m,
                                            drone.comprimento_pegada_m))


def raster_area(livre, max_pontos: int = 200_000):
    passo = max(1.0, math.sqrt(livre.area / max_pontos))
    minx, miny, maxx, maxy = livre.bounds
    xs = np.arange(minx + passo / 2, maxx, passo)
    ys = np.arange(miny + passo / 2, maxy, passo)
    X, Y = np.meshgrid(xs, ys)
    X, Y = X.ravel(), Y.ravel()
    dentro = shapely.contains_xy(livre, X, Y)
    return X[dentro], Y[dentro], passo


def cobertura_raster(livre, pegadas: list, max_pontos: int = 200_000) -> dict:
    """Fracao de celulas do raster da area livre cobertas por >=1 pegada e por >=2 drones."""
    X, Y, passo = raster_area(livre, max_pontos)
    cont = np.zeros(len(X), dtype=np.int16)
    for g in pegadas:
        if g is None or g.is_empty:
            continue
        shapely.prepare(g)
        cont += shapely.contains_xy(g, X, Y).astype(np.int16)
    n = max(1, len(X))
    return {"cobertura_pct": 100.0 * float((cont >= 1).sum()) / n,
            "sobreposicao_entre_drones_pct": 100.0 * float((cont >= 2).sum()) / n,
            "raster_passo_m": passo, "raster_celulas": int(n)}


def resumo_plano(area, frota: list[Drone], rotas: list[Rota], missoes: list[ResultadoMissao],
                 regioes: list, pesos: list) -> dict:
    tempos = np.array([m.tempo_total_s for m in missoes])
    dist = np.array([r.distancia_m for r in rotas])
    curvas = np.array([r.n_curvas for r in rotas])
    areas = np.array([g.area if g is not None else 0.0 for g in regioes])
    pegadas = [pegada_rota(r, d) for r, d in zip(rotas, frota)]
    cob = cobertura_raster(area.livre, pegadas)
    alvo = np.asarray(pesos) * area.livre.area
    return {
        "makespan_s": float(tempos.max()),
        "tempo_medio_s": float(tempos.mean()),
        "cv_tempo": float(tempos.std() / tempos.mean()) if tempos.mean() > 0 else 0.0,
        "cv_area": float(areas.std() / areas.mean()) if areas.mean() > 0 else 0.0,
        "erro_area_vs_alvo_max_pct": float(100 * np.max(np.abs(areas - alvo) / alvo)),
        "distancia_total_m": float(dist.sum()),
        "curvas_total": int(curvas.sum()),
        "n_voos_total": int(sum(m.n_voos for m in missoes)),
        "todos_viaveis_um_voo": bool(all(m.viavel_um_voo for m in missoes)),
        "regioes_desconexas": int(sum(1 for g in regioes if g is not None and g.geom_type == "MultiPolygon")),
        **cob,
        "por_drone": [{"id": d.id, "tipo": d.tipo, "area_m2": float(a), "tempo_s": float(m.tempo_total_s),
                       "distancia_m": float(r.distancia_m), "curvas": int(r.n_curvas), "voos": int(m.n_voos),
                       "angulo_graus": float(r.angulo), "espacamento_m": float(r.info.get('espacamento_real', r.espacamento)),
                       "uso_bateria_pct_1o_voo": float(100 * min(m.tempo_voo_s, d.autonomia_util_s) / (d.autonomia_min * 60))}
                      for d, a, m, r in zip(frota, areas, missoes, rotas)],
    }
