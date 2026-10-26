"""Diagnostico (sem selecao de parametros): erro da contagem de pessoas unicas em funcao do comprimento minimo de
trilha L, no val (onde L foi escolhido) e no test-dev, para a combinacao principal e a referencia COCO.
Le resultados/metricas_por_sequencia.json (que guarda a contagem para todos os L da grade).
Uso (roda no Windows ou no container): python 52_erro_unico_vs_L.py <pasta validacao_videos>
Saida: evidencias/v13_erro_unico_vs_L.png e resultados/erro_unico_vs_L.json
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B = sys.argv[1] if len(sys.argv) > 1 else "/w/validacao_videos"
ps = json.load(open(f"{B}/resultados/metricas_por_sequencia.json"))
meta = json.load(open(f"{B}/resultados/evidencias_meta.json"))
sel = json.load(open(f"{B}/resultados/selecao_val.json"))
LS = sel["Ls"]
NOME = {"ft960": "YOLO11s ajustado 960", "ft1280": "YOLO11s ajustado 1280", "coco960": "YOLO11s COCO 960",
        "coco1280": "YOLO11s COCO 1280", "bytetrack": "ByteTrack", "botsort": "BoT-SORT"}
out = {}
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for combo, cor in ((meta["principal"], "#1f77b4"), (meta["referencia"], "0.45")):
    v, n = combo.split("|")
    for split, ls in (("val", "--"), ("test-dev", "-")):
        rs = [r for r in ps if r["variante"] == v and r["rastreador"] == n and r["split"] == split
              and r["rotulo"] == "escolhida_val" and r["unicos_gt"] > 0]
        eabs = [float(np.mean([abs(r["unicos_pred"][str(L)] - r["unicos_gt"]) / r["unicos_gt"] for r in rs])) for L in LS]
        etot = [float(sum(r["unicos_pred"][str(L)] for r in rs) / sum(r["unicos_gt"] for r in rs) - 1) for L in LS]
        out[f"{combo}|{split}"] = {"L": LS, "erro_rel_abs_medio": eabs, "erro_rel_total": etot}
        axs[0].plot(LS, 100 * np.array(eabs), ls, color=cor, marker=".", label=f"{NOME[v]} + {NOME[n]} — {split}")
        axs[1].plot(LS, 100 * np.array(etot), ls, color=cor, marker=".", label=f"{NOME[v]} + {NOME[n]} — {split}")
    axs[0].axvline(sel["escolha"][combo]["L"], color=cor, lw=0.8, ls=":")
axs[0].set_xscale("log"); axs[1].set_xscale("log")
axs[0].set_xlabel("comprimento mínimo da trilha L (quadros)"); axs[1].set_xlabel("comprimento mínimo da trilha L (quadros)")
axs[0].set_ylabel("erro relativo absoluto médio por sequência (%)"); axs[1].set_ylabel("erro relativo do total de pessoas (%)")
axs[1].axhline(0, color="k", lw=0.7)
axs[0].set_title("Pessoas únicas: erro por sequência × L\n(pontilhado vertical = L escolhido no val)")
axs[1].set_title("Pessoas únicas: viés do total somado × L")
axs[0].legend(fontsize=7); axs[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{B}/evidencias/v13_erro_unico_vs_L.png", dpi=150)
json.dump(out, open(f"{B}/resultados/erro_unico_vs_L.json", "w"), indent=1)
for k, d in out.items():
    print(k, " ".join(f"L{L}:{100*e:.0f}%/{100*t:+.0f}%" for L, e, t in zip(d["L"], d["erro_rel_abs_medio"], d["erro_rel_total"])))
