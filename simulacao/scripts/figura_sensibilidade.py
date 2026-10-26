"""Figura da sensibilidade da agregacao (resultados/sensibilidade_agregacao.json)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SIM = Path(__file__).resolve().parents[1]
ent = Path(sys.argv[1]) if len(sys.argv) > 1 else SIM / "resultados" / "sensibilidade_agregacao.json"
sai = Path(sys.argv[2]) if len(sys.argv) > 2 else SIM / "evidencias" / "sim_04_sensibilidade_agregacao.png"
L = json.loads(ent.read_text(encoding="utf-8"))["linhas"]
COR = {"corretos": "#2a78d6", "duplicatas": "#eb6834", "falsos": "#1baf7a"}
TXT, TXT2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"
dets = [("ideal", "detector ideal"), ("p0.8_s1_fp0.3", "revocacao 0,8; ruido 1 px; 0,3 FP/quadro"),
        ("p0.6_s2_fp1.0", "revocacao 0,6; ruido 2 px; 1 FP/quadro")]
cfgs = sorted({(r["gap_max_andando_s"], r["v_assoc_ms"]) for r in L})
fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
for ax, (det, titulo) in zip(axs, dets):
    agg = defaultdict(lambda: defaultdict(list))
    for r in L:
        if r["detector"] == det:
            k = (r["gap_max_andando_s"], r["v_assoc_ms"])
            for c in ("clusters_corretos_unicos", "clusters_duplicados", "clusters_falsos", "pessoas_vistas_gabarito", "cerise_drones_estatico"):
                agg[k][c].append(r[c])
    x = np.arange(len(cfgs))
    cor = np.array([np.mean(agg[k]["clusters_corretos_unicos"]) for k in cfgs])
    dup = np.array([np.mean(agg[k]["clusters_duplicados"]) for k in cfgs])
    fal = np.array([np.mean(agg[k]["clusters_falsos"]) for k in cfgs])
    vis = np.mean([np.mean(agg[k]["pessoas_vistas_gabarito"]) for k in cfgs])
    est = np.array([np.mean(agg[k]["cerise_drones_estatico"]) for k in cfgs])
    w = 0.62
    ax.bar(x, cor, w, color=COR["corretos"], label="clusters corretos (pessoas distintas)", edgecolor="white", linewidth=1)
    ax.bar(x, dup, w, bottom=cor, color=COR["duplicatas"], label="duplicatas", edgecolor="white", linewidth=1)
    ax.bar(x, fal, w, bottom=cor + dup, color=COR["falsos"], label="falsos positivos", edgecolor="white", linewidth=1)
    for i in range(len(x)):
        tot = cor[i] + dup[i] + fal[i]
        ax.text(x[i], tot + 1.2, f"{tot:.0f}", ha="center", va="bottom", fontsize=8.5, color=TXT)
        ax.text(x[i], cor[i] / 2, f"{cor[i]:.0f}", ha="center", va="center", fontsize=7.5, color="white")
    ax.axhline(vis, color=TXT2, ls="--", lw=1.2, label=f"pessoas distintas vistas no gabarito ({vis:.0f})")
    ax.plot(x + 0.42, est, "D", color=TXT, ms=5, label="agregacao estatica do cerise_drones (contagem sem rotulo)")
    ax.set_xticks(x, [f"{g:.0f} s\n{v:.1f} m/s" for g, v in cfgs], fontsize=8)
    ax.set_title(titulo, fontsize=9.5, loc="left", color=TXT)
    ax.set_xlabel("intervalo max. p/ reassociar quem anda / incerteza de velocidade", fontsize=8, color=TXT2)
    ax.grid(axis="y", color=GRADE, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(0, 105)
axs[0].set_ylabel("contagem unica global (pessoas)", color=TXT2)
h, lab = axs[0].get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", ncol=5, fontsize=8.2, frameon=False)
fig.suptitle("Sensibilidade da agregacao espaco-temporal - cenario reduzido (80 pessoas, 4 drones), execucao sintetica sem render; "
             "media de 3 repeticoes nos detectores ruidosos", x=0.01, ha="left", fontsize=10.5)
fig.tight_layout(rect=(0, 0.07, 1, 0.95))
sai.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(sai, dpi=150)
print(sai)
