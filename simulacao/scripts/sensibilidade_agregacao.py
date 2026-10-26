"""Sensibilidade da agregacao espaco-temporal (sem render, detector oraculo) no cenario reduzido.

Mede, para combinacoes de (gap_max_andando_s, v_assoc_ms) e de qualidade do detector oraculo, a contagem
unica global e a sua decomposicao (corretos, duplicatas, falsos, perdidas) contra o gabarito.
Uso: python scripts/sensibilidade_agregacao.py --execucao cenario/reduzido/sem_render --cenario cenario/reduzido
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM)); sys.path.insert(0, str(SIM.parent))
from shapely.geometry import MultiPolygon, Polygon  # noqa: E402

from cerise_sim.avaliacao import avaliar_contagem  # noqa: E402
from cerise_sim.contagem import DetectorOraculo, ParamsContagem, contar_execucao  # noqa: E402
from cerise_sim.dados import Execucao  # noqa: E402


def poli(lst):
    return Polygon(lst[0]) if len(lst) == 1 else MultiPolygon([Polygon(c) for c in lst])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execucao", required=True)
    ap.add_argument("--cenario", required=True)
    ap.add_argument("--saida", default=str(SIM / "resultados" / "sensibilidade_agregacao.json"))
    ap.add_argument("--repeticoes", type=int, default=3)
    a = ap.parse_args()
    cen = json.loads((Path(a.cenario) / "cenario.json").read_text(encoding="utf-8"))
    ex = Execucao(a.execucao)
    ids = [d for d in cen["regioes_enu"] if d in ex.cameras("drone")]
    reg = [poli(cen["regioes_enu"][d]) for d in ids]
    area = poli(cen["area_enu"])
    gts = {c: ex.caixas_gt(c) for c in ids}
    detectores = {"ideal": dict(p_det=1.0), "p0.8_s1_fp0.3": dict(p_det=0.8, sigma_px=1.0, fp_por_quadro=0.3),
                  "p0.6_s2_fp1.0": dict(p_det=0.6, sigma_px=2.0, fp_por_quadro=1.0)}
    grade = list(itertools.product([30.0, 60.0, 120.0], [0.3, 0.6]))
    linhas = []
    t0 = time.time()
    for (gap, vas), (nome, kw) in itertools.product(grade, detectores.items()):
        reps = 1 if nome == "ideal" else a.repeticoes
        for r in range(reps):
            p = ParamsContagem(gap_max_andando_s=gap, v_assoc_ms=vas)

            def fab(cam, kw=kw, r=r):
                K = ex.intrinsecos(cam)
                o = DetectorOraculo(gts[cam], K.largura, K.altura, seed=100 * r + sum(map(ord, cam)), **kw)
                return lambda q: o(q.frame)
            res = contar_execucao(ex, fab, area, reg, ids, p, por_frame=True)
            av = avaliar_contagem(res, gts, len(ex.pessoas()))
            cd = res["agregacao"].get("cerise_drones_agregacao", {})
            linhas.append({"gap_max_andando_s": gap, "v_assoc_ms": vas, "detector": nome, "rep": r,
                           **{k: av[k] for k in ("pessoas_na_cena", "pessoas_vistas_gabarito", "contagem_proposta", "erro_pct_vs_vistas",
                                                 "clusters_corretos_unicos", "clusters_duplicados", "clusters_falsos", "pessoas_perdidas")},
                           "faixas_sem_posse": res["agregacao"]["faixas_sem_posse"], "soma_deteccoes": res["agregacao"]["soma_deteccoes"],
                           "cerise_drones_estatico": cd.get("proposto_posse_dedup_borda")})
            print(json.dumps(linhas[-1]), flush=True)
    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida).write_text(json.dumps({"execucao": str(a.execucao), "tempo_s": time.time() - t0, "linhas": linhas}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
