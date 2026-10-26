"""Figura com os 4 cenarios do benchmark planejados com DARP + refinamento (N=5, frota heterogenea)."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from cerise_drones.cenarios import CENARIOS, frota_padrao  # noqa: E402
from cerise_drones.planejador import planejar_missao  # noqa: E402
from cerise_drones.visualizar import CORES, desenhar_plano, estilo  # noqa: E402

estilo()
fig, axs = plt.subplots(2, 2, figsize=(11.5, 9.2))
for ax, cen in zip(axs.ravel(), ["ufg", "convexo", "concavo_L", "concavo_U"]):
    area = CENARIOS[cen]()
    frota = frota_padrao(area, 5, True)
    p = planejar_missao(area, frota, "darp", refinar=4, sementes="centros")
    m = p.metricas
    desenhar_plano(ax, area, frota, p.particao.regioes, p.rotas, None, legenda=False)
    ax.set_title(f"{area.nome} ({area.livre.area / 1e4:.0f} ha)\nmakespan {m['makespan_s'] / 60:.1f} min, CV {m['cv_tempo']:.3f}, "
                 f"cobertura {m['cobertura_pct']:.1f}%, {m['n_voos_total']} voos", loc="left", fontsize=8.5)
    ax.tick_params(labelsize=7)
tipos = {"A": "tipo A: 10 m/s, 20 min", "B": "tipo B: 6 m/s, 30 min", "C": "tipo C: 14 m/s, 25 min, grande-angular"}
h = [Patch(facecolor=CORES[i], alpha=0.5, label=f"D{i + 1} ({tipos[t]})") for i, t in enumerate(["A", "B", "C", "A", "B"])]
h.append(Patch(facecolor="#d03b3b", alpha=0.35, hatch="////", label="zona proibida (linha tracejada = margem de 10 m)"))
fig.legend(handles=h, loc="lower center", ncol=3, fontsize=7.5)
fig.suptitle("Cenarios do benchmark - DARP + refinamento, 5 drones heterogeneos (triangulos = bases)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0.06, 1, 0.97))
fig.savefig(RAIZ / "evidencias" / "02b_cenarios_darp_ref.png", bbox_inches="tight")
