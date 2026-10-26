"""Simulacao da agregacao de contagens entre drones (pessoas sinteticas no Campus Samambaia).

As pessoas sao pontos sinteticos (estaticos durante o voo); cada foto (disparo a cada
distancia_disparo_m ao longo das pernas de cobertura) detecta as pessoas dentro da pegada com
probabilidade = revocacao do detector e gera falsos positivos coerentes com a precisao medida.
Erros de georreferenciamento: ruido por deteccao (sigma_det) + deslocamento por foto (sigma_foto).

Uso: python scripts/agregacao_sim.py --revocacao 0.45 --precisao 0.60
"""
import argparse
import json
import math
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
import shapely  # noqa: E402

from cerise_drones.agregacao import contar  # noqa: E402
from cerise_drones.cenarios import area_ufg, frota_padrao  # noqa: E402
from cerise_drones.planejador import planejar_missao  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--revocacao", type=float, default=0.5)
ap.add_argument("--precisao", type=float, default=0.6)
ap.add_argument("--sigma_det", type=float, default=0.8)
ap.add_argument("--sigma_foto", type=float, default=0.8)
ap.add_argument("--n_pessoas", type=int, default=300)
ap.add_argument("--repeticoes", type=int, default=20)
a = ap.parse_args()

area = area_ufg()
frota = frota_padrao(area, 4, heterogenea=True)
plano = planejar_missao(area, frota, "darp", refinar=4, sementes="centros")
regioes = plano.particao.regioes
fronteiras = shapely.union_all([g.boundary for g in regioes]).difference(area.livre.boundary.buffer(0.5))

# fotos: (centro, direcao, drone) ao longo das pernas de cobertura
fotos = []
for i, (d, r) in enumerate(zip(frota, plano.rotas)):
    passo = d.distancia_disparo_m(plano.sobreposicao_frontal)
    for (p0, p1) in r.pernas_cobertura():
        p0 = np.array(p0); p1 = np.array(p1)
        L = float(np.linalg.norm(p1 - p0))
        u = (p1 - p0) / L if L > 1e-6 else np.array([1.0, 0.0])
        for s in np.arange(0, L + 1e-6, passo) if L > 0 else [0.0]:
            fotos.append((p0 + u * s, u, i))
print("fotos simuladas:", len(fotos))

rng = np.random.default_rng(0)
res = []
for rep in range(a.repeticoes):
    # pessoas: uniformes na area + grupos de 8 pessoas sobre as fronteiras entre subareas
    minx, miny, maxx, maxy = area.livre.bounds
    pts = []
    while len(pts) < a.n_pessoas:
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if area.livre.contains(shapely.Point(x, y)):
            pts.append((x, y))
    for k in range(6):
        c = fronteiras.interpolate(rng.uniform(0, fronteiras.length))
        for _ in range(8):
            pts.append((c.x + rng.normal(0, 3), c.y + rng.normal(0, 3)))
    P = np.array(pts)
    det, dono = [], []
    tp_tot = 0
    for (c, u, i) in fotos:
        d = frota[i]
        n = np.array([-u[1], u[0]])
        rel = P - c
        a_long = rel @ u; a_trans = rel @ n
        dentro = (np.abs(a_long) <= d.comprimento_pegada_m / 2) & (np.abs(a_trans) <= d.largura_pegada_m / 2)
        vis = np.nonzero(dentro)[0]
        detect = vis[rng.random(len(vis)) < a.revocacao]
        off = rng.normal(0, a.sigma_foto, 2)
        for j in detect:
            det.append(P[j] + off + rng.normal(0, a.sigma_det, 2)); dono.append(i)
        tp_tot += len(detect)
        # falsos positivos: E[FP] = TP*(1-P)/P, posicoes uniformes na pegada (nao se repetem entre fotos)
        nfp = rng.poisson(len(vis) * a.revocacao * (1 - a.precisao) / a.precisao)
        for _ in range(nfp):
            al = rng.uniform(-d.comprimento_pegada_m / 2, d.comprimento_pegada_m / 2)
            at = rng.uniform(-d.largura_pegada_m / 2, d.largura_pegada_m / 2)
            det.append(c + u * al + n * at); dono.append(i)
    c = contar(np.array(det), np.array(dono), regioes, raio_m=2.0, banda_m=3.0)
    c["verdade"] = len(P)
    res.append(c)
verdade = np.array([r["verdade"] for r in res])
saida = {"parametros": vars(a), "fotos": len(fotos), "repeticoes": a.repeticoes}
for k in ("soma_ingenua", "dedup_por_drone_sem_posse", "proposto_posse_dedup_borda"):
    v = np.array([r[k] for r in res])
    e = 100 * (v - verdade) / verdade
    # contagem corrigida pela revocacao/precisao do detector: N_est = N * P / R_efetiva (R_efetiva ~ prob.
    # de ser detectado em pelo menos uma das k fotos que veem a pessoa) - aqui so reportado o bruto
    saida[k] = {"media": float(v.mean()), "erro_pct_medio": float(e.mean()), "erro_pct_desvio": float(e.std())}
saida["frac_deteccoes_na_borda"] = float(np.mean([r["frac_deteccoes_na_borda"] for r in res]))
saida["verdade_media"] = float(verdade.mean())
print(json.dumps(saida, indent=1))
json.dump(saida, open(RAIZ / "resultados" / "agregacao_sim.json", "w"), indent=1)
