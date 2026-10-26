"""Conferencia da implementacao das metricas (metricas_mot.py):
  (1) testes sinteticos com resultado conhecido (rastreador perfeito, troca de ID, falsos positivos);
  (2) MOTA, IDF1 e IDSW conferidos contra o py-motmetrics 1.4.0 (PyPI) nas trilhas finais (rastreios/*),
      sequencia a sequencia. A matriz de distancias do motmetrics e montada aqui (1 - IoU, NaN se IoU < 0,5)
      porque motmetrics.distances.iou_matrix usa np.asfarray, removido no numpy 2.
Saida: resultados/conferencia_metricas.json
"""
import glob
import json
import os

import motmetrics
import numpy as np

import metricas_mot as mm

BASE = "/w/validacao_videos"
MOT = "/w/mot"
res = {"sinteticos": {}, "motmetrics": []}


def caixa(x, y, s=10):
    return [x, y, x + s, y + s]


# ---------- (1) testes sinteticos
T = 20
gt = {t: (np.array([1, 2]), np.array([caixa(10 + t, 10), caixa(100, 50 + t)], float)) for t in range(1, T + 1)}
perfeito = mm.resumir([mm.avaliar(mm.montar_sequencia(gt, gt, T))])
troca = {t: (np.array([7, 8]) if t <= 10 else np.array([7, 9]), gt[t][1]) for t in range(1, T + 1)}
r_troca = mm.resumir([mm.avaliar(mm.montar_sequencia(gt, troca, T))])
fp = {t: (np.r_[gt[t][0], 99], np.r_[gt[t][1], [caixa(300, 300)]]) for t in range(1, T + 1)}
r_fp = mm.resumir([mm.avaliar(mm.montar_sequencia(gt, fp, T))])
res["sinteticos"] = {
    "perfeito": {k: perfeito[k] for k in ("HOTA", "MOTA", "IDF1", "IDSW")},
    "uma_troca_de_id": {k: r_troca[k] for k in ("HOTA", "AssA", "DetA", "MOTA", "IDF1", "IDSW")},
    "um_fp_por_quadro": {k: r_fp[k] for k in ("HOTA", "DetA", "MOTA", "IDF1", "IDSW")},
}
# valores esperados analiticamente
assert abs(perfeito["HOTA"] - 1) < 1e-9 and abs(perfeito["MOTA"] - 1) < 1e-9 and abs(perfeito["IDF1"] - 1) < 1e-9
assert r_troca["IDSW"] == 1 and abs(r_troca["MOTA"] - (40 - 1) / 40) < 1e-9 and abs(r_troca["IDF1"] - 30 / 40) < 1e-9
assert abs(r_fp["MOTA"] - (40 - 20) / 40) < 1e-9 and abs(r_fp["IDF1"] - 40 / 50) < 1e-9
# troca de ID: AssA esperado = media ponderada de |TPA|/(|TPA|+|FNA|+|FPA|): 20 det. do id1 com 1,0 e 10+10 do id2 com 0,5
assert abs(r_troca["AssA"] - (20 * 1 + 20 * 0.5) / 40) < 1e-9
print("testes sinteticos OK", json.dumps(res["sinteticos"]), flush=True)


# ---------- (2) motmetrics
def motmetrics_seq(gtq, trq, T):
    acc = motmetrics.MOTAccumulator(auto_id=True)
    for t in range(1, T + 1):
        gi, gb = gtq[t]; ti, tb = trq.get(t, (np.zeros(0, int), np.zeros((0, 4))))
        d = 1 - mm.iou(gb, tb)
        d[d > 0.5] = np.nan
        acc.update(gi.tolist(), ti.tolist(), d)
    mh = motmetrics.metrics.create()
    s = mh.compute(acc, metrics=["mota", "idf1", "num_switches", "num_false_positives", "num_misses"], name="x")
    return {k: float(s[k].iloc[0]) for k in s.columns}


for pasta in sorted(glob.glob(f"{BASE}/rastreios/*/test-dev")) + sorted(glob.glob(f"{BASE}/rastreios/*/val")):
    combo = pasta.split("/")[-2]; split = pasta.split("/")[-1]
    if combo not in ("ft960_bytetrack", "ft960_botsort", "coco960_bytetrack"):
        continue
    for arq in sorted(glob.glob(f"{pasta}/*.txt")):
        seq = os.path.basename(arq)[:-4]
        T = len(os.listdir(f"{MOT}/VisDrone2019-MOT-{split}/sequences/{seq}"))
        g, ign = mm.ler_gabarito(f"{MOT}/VisDrone2019-MOT-{split}/annotations/{seq}.txt", T)
        gtq = {t: mm.filtrar_ignorado(g[t][0], g[t][1], ign[t]) for t in range(1, T + 1)}
        d = np.loadtxt(arq, delimiter=",", ndmin=2)
        trq = {}
        for t in range(1, T + 1):
            m = d[:, 0] == t if len(d) else np.zeros(0, bool)
            x = d[m] if len(d) else np.zeros((0, 10))
            trq[t] = (x[:, 1].astype(int), np.c_[x[:, 2], x[:, 3], x[:, 2] + x[:, 4], x[:, 3] + x[:, 5]])
        nosso = mm.resumir([mm.avaliar(mm.montar_sequencia(gtq, trq, T))])
        ref = motmetrics_seq(gtq, trq, T)
        lin = {"combo": combo, "split": split, "seq": seq, "MOTA_nosso": nosso["MOTA"], "MOTA_motmetrics": ref["mota"],
               "IDF1_nosso": nosso["IDF1"], "IDF1_motmetrics": ref["idf1"], "IDSW_nosso": nosso["IDSW"],
               "IDSW_motmetrics": ref["num_switches"], "FP_nosso": nosso["CLR_FP"], "FP_motmetrics": ref["num_false_positives"],
               "FN_nosso": nosso["CLR_FN"], "FN_motmetrics": ref["num_misses"]}
        res["motmetrics"].append(lin)
        print(json.dumps(lin), flush=True)

if res["motmetrics"]:
    dm = [abs(x["MOTA_nosso"] - x["MOTA_motmetrics"]) for x in res["motmetrics"] if np.isfinite(x["MOTA_motmetrics"])]
    di = [abs(x["IDF1_nosso"] - x["IDF1_motmetrics"]) for x in res["motmetrics"] if np.isfinite(x["IDF1_motmetrics"])]
    res["resumo"] = {"n_seq": len(res["motmetrics"]), "max_dif_MOTA": max(dm), "max_dif_IDF1": max(di),
                     "media_dif_MOTA": float(np.mean(dm)), "media_dif_IDF1": float(np.mean(di))}
    print("RESUMO", json.dumps(res["resumo"]), flush=True)
json.dump(res, open(f"{BASE}/resultados/conferencia_metricas.json", "w"), indent=1, default=float)
print("FIM_CONFERENCIA", flush=True)
