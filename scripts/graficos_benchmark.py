"""Graficos e tabelas do benchmark de particao e da ablacao.

Uso: python scripts/graficos_benchmark.py
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cerise_drones.visualizar import TINTA, TINTA2, estilo  # noqa: E402

RES = RAIZ / "resultados"; EV = RAIZ / "evidencias"
df = pd.read_csv(RES / "benchmark.csv")
METODOS = ["faixas", "kmeans", "darp", "darp+ref"]
ROT = {"faixas": "Faixas ponderadas", "kmeans": "k-means + Voronoi", "darp": "DARP", "darp+ref": "DARP + refinamento"}
COR = {"faixas": "#2a78d6", "kmeans": "#eb6834", "darp": "#1baf7a", "darp+ref": "#6250d6"}
MARK = {"faixas": "o", "kmeans": "s", "darp": "^", "darp+ref": "D"}
_ha = df.groupby("cenario").area_livre_m2.first() / 1e4
CEN = {"ufg": f"UFG Samambaia ({_ha['ufg']:.0f} ha, real)", "convexo": f"Convexo ({_ha['convexo']:.0f} ha)",
       "concavo_L": f"Concavo L + 1 zona ({_ha['concavo_L']:.0f} ha)", "concavo_U": f"Concavo U + 2 zonas ({_ha['concavo_U']:.0f} ha)"}
estilo()

# ---------------- makespan x N (small multiples: linhas = frota, colunas = cenario)
fig, axs = plt.subplots(2, 4, figsize=(14, 6.4), sharex=True)
for r, fr in enumerate(["homogenea", "heterogenea"]):
    for c, cen in enumerate(CEN):
        ax = axs[r, c]
        sub = df[(df.cenario == cen) & (df.frota == fr)]
        for m in METODOS:
            s = sub[sub.metodo == m].sort_values("n")
            ax.plot(s.n, s.makespan_s / 60, color=COR[m], marker=MARK[m], ms=5, lw=2, label=ROT[m])
        if r == 0:
            ax.set_title(CEN[cen], loc="left", fontsize=9, color=TINTA)
        if c == 0:
            ax.set_ylabel(f"frota {fr}\nmakespan (min)")
        if r == 1:
            ax.set_xlabel("numero de drones (N)")
        ax.set_xticks([2, 3, 4, 5, 6])
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5)
fig.suptitle("Makespan (maior tempo de missao, inclui subida/descida, curvas e trocas de bateria) por cenario e N",
             x=0.01, ha="left", fontsize=10.5)
fig.tight_layout(rect=(0, 0.05, 1, 0.96))
fig.savefig(EV / "04_benchmark_makespan.png", bbox_inches="tight")
plt.close(fig)

# ---------------- balanceamento, distancia, curvas, tempo de execucao (media sobre cenarios e N)
agg = df.groupby(["frota", "metodo"]).agg(makespan_min=("makespan_s", lambda x: x.mean() / 60),
                                          cv_tempo=("cv_tempo", "mean"), distancia_km=("distancia_total_m", lambda x: x.mean() / 1000),
                                          curvas=("curvas_total", "mean"), cobertura_pct=("cobertura_pct", "mean"),
                                          sobreposicao_pct=("sobreposicao_entre_drones_pct", "mean"),
                                          voos=("n_voos_total", "mean"), tempo_alg_s=("tempo_algoritmo_s", "median"),
                                          tempo_alg_max_s=("tempo_algoritmo_s", "max")).reset_index()
vit = df.loc[df.groupby(["cenario", "n", "frota"])["makespan_s"].idxmin()].groupby(["frota", "metodo"]).size()
agg["vitorias_makespan"] = [int(vit.get((f, m), 0)) for f, m in zip(agg.frota, agg.metodo)]
# makespan relativo ao melhor de cada caso
best = df.groupby(["cenario", "n", "frota"])["makespan_s"].transform("min")
df["mk_rel"] = df.makespan_s / best
agg["makespan_rel_medio"] = [df[(df.frota == f) & (df.metodo == m)].mk_rel.mean() for f, m in zip(agg.frota, agg.metodo)]
agg.to_csv(RES / "benchmark_resumo.csv", index=False)

fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
x = np.arange(2)
wd = 0.19
for k, m in enumerate(METODOS):
    s = agg[agg.metodo == m].set_index("frota").loc[["homogenea", "heterogenea"]]
    for ax, col in zip(axs, ["cv_tempo", "makespan_rel_medio", "curvas", "tempo_alg_s"]):
        v = s[col].values
        ax.bar(x + (k - 1.5) * wd, v, wd * 0.92, color=COR[m], label=ROT[m])
tit = ["(a) CV do tempo de missao (menor = mais balanceado)", "(b) makespan / melhor makespan do caso",
       "(c) numero total de curvas (media)", "(d) tempo de execucao (mediana, s)"]
for ax, t in zip(axs, tit):
    ax.set_xticks(x); ax.set_xticklabels(["frota homogenea", "frota heterogenea"])
    ax.set_title(t, loc="left", fontsize=8.5, color=TINTA)
axs[1].set_ylim(0.95, None)
axs[3].set_yscale("log")
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402
axs[3].yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
axs[3].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
axs[3].yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(EV / "05_benchmark_metricas.png", bbox_inches="tight")
plt.close(fig)

# ---------------- tabelas markdown
def md(dfx, cols, fmt):
    lin = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in dfx.iterrows():
        lin.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |")
    return "\n".join(lin)


fmt = {"frota": "{}", "metodo": "{}", "makespan_min": "{:.2f}", "makespan_rel_medio": "{:.3f}", "cv_tempo": "{:.3f}",
       "distancia_km": "{:.2f}", "curvas": "{:.1f}", "cobertura_pct": "{:.2f}", "sobreposicao_pct": "{:.1f}",
       "voos": "{:.2f}", "tempo_alg_s": "{:.2f}", "vitorias_makespan": "{}"}
cols = list(fmt)
tab = md(agg[cols], cols, fmt)
(RES / "benchmark_resumo.md").write_text(tab, encoding="utf-8")
print(tab)

# tabela detalhada do cenario UFG, frota heterogenea
u = df[(df.cenario == "ufg") & (df.frota == "heterogenea")][["n", "metodo", "makespan_s", "cv_tempo", "distancia_total_m",
                                                               "curvas_total", "cobertura_pct", "n_voos_total", "tempo_algoritmo_s"]].copy()
u["makespan_min"] = u.makespan_s / 60; u["dist_km"] = u.distancia_total_m / 1000
fmt2 = {"n": "{}", "metodo": "{}", "makespan_min": "{:.2f}", "cv_tempo": "{:.3f}", "dist_km": "{:.2f}", "curvas_total": "{}",
        "cobertura_pct": "{:.2f}", "n_voos_total": "{}", "tempo_algoritmo_s": "{:.2f}"}
tab2 = md(u[list(fmt2)], list(fmt2), fmt2)
(RES / "benchmark_ufg_het.md").write_text(tab2, encoding="utf-8")

# ablacao
if (RES / "ablacao.csv").exists():
    ab = pd.read_csv(RES / "ablacao.csv")
    a2 = ab.groupby(["cenario", "metodo"]).agg(makespan_min=("makespan_s", lambda x: x.mean() / 60), cv_tempo=("cv_tempo", "mean"),
                                               erro_area_pct=("erro_area_vs_alvo_max_pct", "mean"),
                                               distancia_km=("distancia_total_m", lambda x: x.mean() / 1000),
                                               voos=("n_voos_total", "mean"), tempo_alg_s=("tempo_algoritmo_s", "median")).reset_index()
    a2.to_csv(RES / "ablacao_resumo.csv", index=False)
    fmt3 = {"cenario": "{}", "metodo": "{}", "makespan_min": "{:.2f}", "cv_tempo": "{:.3f}", "erro_area_pct": "{:.1f}",
            "distancia_km": "{:.2f}", "voos": "{:.2f}", "tempo_alg_s": "{:.2f}"}
    tab3 = md(a2[list(fmt3)], list(fmt3), fmt3)
    (RES / "ablacao_resumo.md").write_text(tab3, encoding="utf-8")
    print(tab3)
