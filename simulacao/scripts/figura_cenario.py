"""Figura do cenario (ENU): terreno procedural, subareas, rotas dos drones, trajetorias das pessoas e
camera de visao geral. Uso: python scripts/figura_cenario.py cenario/reduzido [saida.png]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.patches import Polygon as MPoly  # noqa: E402

CORES = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
COR_TIPO = {"caminhante": "#222222", "grupo_andando": "#e377c2", "grupo_parado": "#ff7f0e", "parado": "#17becf",
            "cruza_fronteira": "#bcbd22"}
COR_MAT = {"grama_seca": "#cfc8a0", "grama": "#9cbf72", "asfalto": "#6b6b6b", "concreto": "#c9c7c0", "terra": "#b89a74"}


def figura(pasta, saida, titulo=None):
    pasta = Path(pasta)
    cen = json.loads((pasta / "cenario.json").read_text(encoding="utf-8"))
    cena = json.loads((pasta / "cena.json").read_text(encoding="utf-8"))
    pes = json.loads((pasta / "pessoas_plano.json").read_text(encoding="utf-8"))["pessoas"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.6), gridspec_kw={"width_ratios": [1.25, 1]})
    for ax in axs:
        for mm in cena["malhas"]:
            V = np.array(mm["vertices"])
            if not len(V):
                continue
            ax.tripcolor(V[:, 0], V[:, 1], np.array(mm["faces"]), facecolors=np.zeros(len(mm["faces"])),
                         cmap=matplotlib.colors.ListedColormap([COR_MAT.get(mm["material"], "#dddddd")]),
                         edgecolors="none", zorder=0)
        for e in cena["edificios"]:
            ax.add_patch(MPoly(e["pegada"], closed=True, fc="#8a8a92", ec="#444", lw=0.5, zorder=1))
        for t in cena["arvores"]:
            ax.add_patch(plt.Circle((t["x"], t["y"]), t["raio_copa_m"], fc="#3d6b35", ec="none", alpha=0.85, zorder=2))
        ax.set_aspect("equal")
        ax.set_xlabel("x leste (m)"); ax.set_ylabel("y norte (m)")
    ax = axs[0]
    for i, d in enumerate(cen["drones"]):
        for poly in cen["regioes_enu"][d["id"]]:
            ax.add_patch(MPoly(poly, closed=True, fc=CORES[i], alpha=0.18, ec=CORES[i], lw=1.2, zorder=3))
        xyz = np.array(d["trajetoria"]["xyz"])
        ax.plot(xyz[:, 0], xyz[:, 1], "-", color=CORES[i], lw=1.1, zorder=4)
        ax.plot(*d["base_enu"], "^", color=CORES[i], ms=9, mec="k", zorder=5)
    vg = cen["visao_geral"]
    ax.plot(vg["posicao"][0], vg["posicao"][1], "s", color="k", ms=7, zorder=6)
    ax.annotate("camera de visao geral", vg["posicao"][:2], xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_title(f"(a) Subareas e rotas (DARP + refinamento), {cen['area_ha']:.1f} ha, {len(cen['drones'])} drones", loc="left", fontsize=10)
    leg = [Line2D([], [], color=CORES[i], lw=2, label=f"{d['id']} tipo {d['tipo']} ({d['velocidade_ms']:.0f} m/s, "
                  f"{d['camera']['intrinsecos']['largura']}x{d['camera']['intrinsecos']['altura']})")
           for i, d in enumerate(cen["drones"])]
    leg += [Line2D([], [], marker="^", ls="", color="gray", mec="k", label="base"),
            Patch(fc="#8a8a92", ec="#444", label="edificio esquematico"), Patch(fc="#3d6b35", label="arvore")]
    ax.legend(handles=leg, fontsize=7.5, loc="lower right")
    ax = axs[1]
    a = np.array(cen["area_enu"][0])
    for p in pes:
        k = np.array(p["kf"])
        c = COR_TIPO.get(p["tipo"], "k")
        ax.plot(k[:, 1], k[:, 2], "-", color=c, lw=0.6, alpha=0.7, zorder=4)
        ax.plot(k[0, 1], k[0, 2], "o", color=c, ms=2.8, zorder=5)
    ax.plot(a[:, 0], a[:, 1], "k-", lw=0.8)
    if cen.get("fronteiras_enu"):
        from shapely.geometry import shape
        g = shape(cen["fronteiras_enu"])
        for ln in getattr(g, "geoms", [g]):
            xs, ys = ln.xy
            ax.plot(xs, ys, "--", color="#555", lw=0.8, zorder=3)
    pad = 15
    ax.set_xlim(a[:, 0].min() - pad, a[:, 0].max() + pad); ax.set_ylim(a[:, 1].min() - pad, a[:, 1].max() + pad)
    cont = {}
    for p in pes:
        cont[p["tipo"]] = cont.get(p["tipo"], 0) + 1
    ax.legend(handles=[Line2D([], [], color=COR_TIPO[t], marker="o", ms=4, lw=1, label=f"{t.replace('_', ' ')} ({n})")
                       for t, n in cont.items()] + [Line2D([], [], color="#555", ls="--", label="fronteira entre subareas")],
              fontsize=7.5, loc="lower right")
    ax.set_title(f"(b) Plano de {len(pes)} pessoas ({cen['duracao_s']:.0f} s): trajetorias e pontos de partida", loc="left", fontsize=10)
    fig.suptitle(titulo or f"Cenario de simulacao '{cen['nome']}' - Campus Samambaia/UFG em coordenadas locais ENU "
                 f"(origem {cen['origem_enu']['lat0']:.5f}, {cen['origem_enu']['lon0']:.5f})", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return saida


if __name__ == "__main__":
    p = Path(sys.argv[1])
    figura(p, sys.argv[2] if len(sys.argv) > 2 else p / "mapa_cenario.png")
