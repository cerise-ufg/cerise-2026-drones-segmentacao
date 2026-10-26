"""Simulacao de falha / bateria baixa no meio da missao e realocacao dinamica (UFG, 4 drones).

Uso: python scripts/colaboracao_sim.py
Saidas: resultados/colaboracao.csv/json, evidencias/06_colaboracao_linha_do_tempo.png,
        evidencias/07_colaboracao_estrategias.png, evidencias/colaboracao_reparticao.gif
"""
import csv
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from cerise_drones.cenarios import area_ufg, frota_padrao  # noqa: E402
from cerise_drones.colaboracao import _pegada, simular_falha  # noqa: E402
from cerise_drones.geo import poligonos  # noqa: E402
from cerise_drones.planejador import planejar_missao  # noqa: E402
from cerise_drones.visualizar import CORES, TINTA, TINTA2, _desenha_geom, estilo  # noqa: E402

EV = RAIZ / "evidencias"; RES = RAIZ / "resultados"
area = area_ufg()
frota = frota_padrao(area, 4, heterogenea=True)
plano = planejar_missao(area, frota, "darp", refinar=4, sementes="centros")
print("makespan base %.1f s" % plano.metricas["makespan_s"])

# ------------------------------------------------------------------ tabela: falho x instante x estrategia
linhas = []
for idx in range(len(frota)):
    for frac in (0.25, 0.5, 0.75):
        for est in ("nenhuma", "reparticao", "leilao"):
            r = simular_falha(plano, idx, frac, est)
            lin = {"falho": r.falho, "frac_missao": frac, "t_falha_s": round(r.t_falha_s, 1), "estrategia": est,
                   "makespan_base_s": round(r.makespan_base_s, 1), "makespan_s": round(r.makespan_s, 1),
                   "aumento_makespan_pct": round(r.aumento_makespan_pct, 2), "area_perdida_m2": round(r.area_perdida_m2, 1),
                   "area_perdida_pct": round(100 * r.area_perdida_m2 / area.livre.area, 2),
                   "cobertura_pct": round(r.cobertura_pct, 2), "tempo_replanejamento_s": round(r.tempo_replanejamento_s, 3),
                   "quadros_lora": r.mensagens["quadros"], "bytes_lora": r.mensagens["bytes"],
                   "tempo_no_ar_SF9_ms": round(r.mensagens["tempo_no_ar_ms"], 1),
                   "por_tipo": json.dumps(r.mensagens["por_tipo"]), **{k: round(v, 1) for k, v in r.detalhes.items()}}
            linhas.append(lin)
            print(lin)
campos = sorted({k for l in linhas for k in l}, key=lambda k: list(linhas[0]).index(k) if k in linhas[0] else 99)
with open(RES / "colaboracao.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=campos)
    w.writeheader(); w.writerows(linhas)
json.dump(linhas, open(RES / "colaboracao.json", "w"), indent=1)

# ------------------------------------------------------------------ caso ilustrativo: D3 (maior subarea) falha a 50%
IDX, FRAC = 2, 0.5
res = {e: simular_falha(plano, IDX, FRAC, e) for e in ("nenhuma", "reparticao", "leilao")}
estilo()
ox, oy = area.area.bounds[0], area.area.bounds[1]


def quadro(ax, r, t, titulo, mostrar_novas=True):
    _desenha_geom(ax, area.area, ox, oy, facecolor="#f0efec", edgecolor="#8a8984", linewidth=0.8)
    for i, g in enumerate(plano.particao.regioes):
        _desenha_geom(ax, g, ox, oy, facecolor="none", edgecolor=CORES[i], linewidth=0.8, alpha=0.6)
    for i, tr in enumerate(r.trajetorias):
        peg = _pegada(tr.pernas_ate(t), frota[i])
        if peg is not None:
            _desenha_geom(ax, peg.intersection(area.livre), ox, oy, facecolor=CORES[i], alpha=0.40, edgecolor="none")
    if mostrar_novas and t >= r.t_falha_s + 3.0:
        sob = [i for i in range(len(frota)) if i != IDX]
        for j, g in enumerate(r.regioes_novas):
            if g is not None and not g.is_empty:
                for p in poligonos(g):
                    xs, ys = zip(*[(x - ox, y - oy) for x, y in p.exterior.coords])
                    ax.plot(xs, ys, color=CORES[sob[j]], lw=1.4, ls="--")
    for i, tr in enumerate(r.trajetorias):
        x, y = tr.pos(min(t, tr.fim))
        if i == IDX and t >= r.t_falha_s:
            ax.plot(x - ox, y - oy, marker="X", ms=11, color="#d03b3b", mec="white", mew=1.0, zorder=7)
        elif t <= tr.fim:
            ax.plot(x - ox, y - oy, marker="o", ms=7, color=CORES[i], mec="white", mew=1.2, zorder=7)
    ax.set_aspect("equal"); ax.autoscale_view()
    ax.set_title(titulo, loc="left", fontsize=8.5, color=TINTA)
    ax.tick_params(labelsize=6.5)


r = res["reparticao"]
tf = r.t_falha_s
tempos = [0.6 * tf, tf + 3.0, tf + 0.5 * (r.makespan_s - tf), r.makespan_s]
rot = ["antes da falha", f"falha de {frota[IDX].id} (t = {tf / 60:.1f} min) + reparticao",
       "sobreviventes cobrem a area orfa", "fim da missao"]
fig, axs = plt.subplots(1, 4, figsize=(15, 4.4))
for ax, t, lb in zip(axs, tempos, rot):
    quadro(ax, r, t, f"t = {t / 60:.1f} min - {lb}")
hand = [Patch(facecolor=CORES[i], alpha=0.45, label=f"coberto por {d.id}") for i, d in enumerate(frota)]
hand += [Line2D([], [], color=TINTA2, ls="--", label="nova subarea (AREA_ASSIGN)"),
         Line2D([], [], marker="X", color="#d03b3b", ls="none", label="drone que falhou")]
fig.legend(handles=hand, loc="lower center", ncol=6, fontsize=7.5)
fig.suptitle(f"Realocacao dinamica por reparticao (DARP) - {frota[IDX].id} abandona a missao a 50% do seu voo; "
             f"makespan {r.makespan_base_s / 60:.1f} -> {r.makespan_s / 60:.1f} min, cobertura {r.cobertura_pct:.1f}%",
             x=0.01, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
fig.savefig(EV / "06_colaboracao_linha_do_tempo.png", bbox_inches="tight")
plt.close(fig)

# comparacao final das 3 estrategias
fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
nomes = {"nenhuma": "(a) sem realocacao", "reparticao": "(b) reparticao do nao coberto (DARP)",
         "leilao": "(c) leilao de tarefas (SSI minimax)"}
for ax, (e, rr) in zip(axs, res.items()):
    quadro(ax, rr, rr.makespan_s, f"{nomes[e]}\nmakespan {rr.makespan_s / 60:.1f} min ({rr.aumento_makespan_pct:+.1f}%), "
                                  f"cobertura {rr.cobertura_pct:.1f}%, {rr.mensagens['quadros']} quadros LoRa")
fig.legend(handles=hand, loc="lower center", ncol=6, fontsize=7.5)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(EV / "07_colaboracao_estrategias.png", bbox_inches="tight")
plt.close(fig)

# GIF da linha do tempo (reparticao)
frames = []
tmp = RES / "_frames"; tmp.mkdir(exist_ok=True)
for k, t in enumerate(np.linspace(0, r.makespan_s, 48)):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    quadro(ax, r, t, f"t = {t / 60:4.1f} min" + ("  -  falha de %s, reparticao" % frota[IDX].id if t >= tf else ""))
    fig.tight_layout()
    fn = tmp / f"f{k:03d}.png"
    fig.savefig(fn, dpi=90)
    plt.close(fig)
    frames.append(imageio.imread(fn))
imageio.mimsave(EV / "colaboracao_reparticao.gif", frames, duration=0.18, loop=0)
for f in tmp.glob("*.png"):
    f.unlink()
tmp.rmdir()
json.dump({e: {"makespan_s": rr.makespan_s, "aumento_pct": rr.aumento_makespan_pct, "cobertura_pct": rr.cobertura_pct,
               "area_perdida_m2": rr.area_perdida_m2, "mensagens": rr.mensagens, "detalhes": rr.detalhes,
               "tempo_replanejamento_s": rr.tempo_replanejamento_s, "t_falha_s": rr.t_falha_s}
           for e, rr in res.items()}, open(RES / "colaboracao_caso_D3_50.json", "w"), indent=1, default=float)
print("ok")
