"""Analise dos erros de deteccao dentro do rastreamento (test-dev, configuracoes escolhidas no val):
  - revocacao por faixa de altura da pessoa no gabarito (px) -> efeito da escala/altitude;
  - natureza dos falsos positivos: para cada caixa do rastreador nao casada (IoU >= 0,5, Hungaro por quadro),
    a maior IoU com qualquer pessoa anotada: >= 0,3 (localizacao/duplicata), 0 < IoU < 0,3 (parcial) ou 0
    (nenhuma pessoa anotada sobreposta: objeto que nao e pessoa ou pessoa nao anotada).
Saida: resultados/analise_erros.json e evidencias/v14_analise_erros.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

import metricas_mot as mm

B = "/w/validacao_videos"; MOT = "/w/mot"
meta = json.load(open(f"{B}/resultados/evidencias_meta.json"))
FAIXAS = [(0, 20), (20, 40), (40, 80), (80, 10000)]
NOME = {"ft960": "YOLO11s ajustado 960", "ft1280": "YOLO11s ajustado 1280", "coco960": "YOLO11s COCO 960",
        "coco1280": "YOLO11s COCO 1280", "bytetrack": "ByteTrack", "botsort": "BoT-SORT"}
res = {}
for combo in (meta["principal"], meta["referencia"]):
    v, n = combo.split("|")
    tp_f = np.zeros(len(FAIXAS)); tot_f = np.zeros(len(FAIXAS))
    fp_cat = {"localizacao_ou_duplicata": 0, "parcial": 0, "sem_pessoa_anotada": 0}
    n_trk = 0
    pasta = f"{B}/rastreios/{v}_{n}/test-dev"
    for arq in sorted(os.listdir(pasta)):
        seq = arq[:-4]
        T = len(os.listdir(f"{MOT}/VisDrone2019-MOT-test-dev/sequences/{seq}"))
        g, ign = mm.ler_gabarito(f"{MOT}/VisDrone2019-MOT-test-dev/annotations/{seq}.txt", T)
        d = np.loadtxt(f"{pasta}/{arq}", delimiter=",", ndmin=2)
        for t in range(1, T + 1):
            gi, gb = mm.filtrar_ignorado(g[t][0], g[t][1], ign[t])
            x = d[d[:, 0] == t] if len(d) else np.zeros((0, 10))
            tb = np.c_[x[:, 2], x[:, 3], x[:, 2] + x[:, 4], x[:, 3] + x[:, 5]]
            n_trk += len(tb)
            s = mm.iou(gb, tb)
            casg = np.zeros(len(gb), bool); cast = np.zeros(len(tb), bool)
            if len(gb) and len(tb):
                r, c = linear_sum_assignment(-s)
                ok = s[r, c] >= 0.5
                casg[r[ok]] = True; cast[c[ok]] = True
            h = gb[:, 3] - gb[:, 1]
            for k, (a, b) in enumerate(FAIXAS):
                m = (h >= a) & (h < b)
                tot_f[k] += m.sum(); tp_f[k] += (m & casg).sum()
            for j in np.flatnonzero(~cast):
                mx = s[:, j].max() if len(gb) else 0.0
                fp_cat["localizacao_ou_duplicata" if mx >= 0.3 else ("parcial" if mx > 0 else "sem_pessoa_anotada")] += 1
    nfp = sum(fp_cat.values())
    res[combo] = {"faixas_altura_px": FAIXAS, "pessoas_por_faixa": tot_f.tolist(), "revocacao_por_faixa": (tp_f / np.maximum(1, tot_f)).tolist(),
                  "caixas_rastreador": int(n_trk), "falsos_positivos": int(nfp),
                  "fp_por_categoria": fp_cat, "fp_fracao": {k: val / max(1, nfp) for k, val in fp_cat.items()}}
    print(combo, json.dumps(res[combo]), flush=True)
json.dump(res, open(f"{B}/resultados/analise_erros.json", "w"), indent=1)

fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
x = np.arange(len(FAIXAS)); w = 0.38
for i, (combo, cor) in enumerate(((meta["principal"], "#1f77b4"), (meta["referencia"], "0.55"))):
    v, n = combo.split("|")
    axs[0].bar(x + (i - 0.5) * w, res[combo]["revocacao_por_faixa"], w, color=cor, label=f"{NOME[v]} + {NOME[n]}")
    cats = list(res[combo]["fp_fracao"].keys())
    axs[1].bar(np.arange(3) + (i - 0.5) * w, [res[combo]["fp_por_categoria"][c] for c in cats], w, color=cor,
               label=f"{NOME[v]} + {NOME[n]} ({res[combo]['falsos_positivos']} FP)")
tot = res[meta["principal"]]["pessoas_por_faixa"]
axs[0].set_xticks(x); axs[0].set_xticklabels([f"{a}–{b if b < 10000 else '∞'} px\n({int(t)} caixas)" for (a, b), t in zip(FAIXAS, tot)])
axs[0].set_ylabel("revocação (IoU ≥ 0,5)"); axs[0].set_title("Revocação por altura da pessoa no gabarito (test-dev)"); axs[0].legend(fontsize=7)
axs[1].set_xticks(np.arange(3)); axs[1].set_xticklabels(["IoU máx. ≥ 0,3\n(localização/duplicata)", "0 < IoU máx. < 0,3\n(parcial)", "IoU = 0\n(sem pessoa anotada)"])
axs[1].set_ylabel("caixas falso-positivas (soma dos quadros)"); axs[1].set_title("Natureza dos falsos positivos do rastreador (test-dev)")
axs[1].set_ylim(0, 1.3 * max(max(r["fp_por_categoria"].values()) for r in res.values()))
axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{B}/evidencias/v14_analise_erros.png", dpi=150)
print("FIM_ANALISE", flush=True)
