"""Demonstracao no Campus Samambaia (UFG): particao entre 4 drones heterogeneos, rotas, exportacao
QGC/KML/GeoJSON, figura PNG, mapa folium e captura com Playwright.

Uso: python scripts/demo_ufg.py
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import matplotlib.pyplot as plt  # noqa: E402

from cerise_drones.cenarios import area_ufg, frota_padrao  # noqa: E402
from cerise_drones.planejador import planejar_missao  # noqa: E402
from cerise_drones.visualizar import (capturar_html, desenhar_plano, estilo, figura_plano,  # noqa: E402
                                      mapa_folium)

EV = RAIZ / "evidencias"
RES = RAIZ / "resultados"
EXP = RAIZ / "resultados" / "missao_ufg"
EV.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)

area = area_ufg()
frota = frota_padrao(area, 4, heterogenea=True)
print("Area livre: %.2f ha, EPSG:%d" % (area.livre.area / 1e4, area.proj.epsg))
for d in frota:
    print(d.id, d.tipo, d.rotulo, "v=%.0f m/s" % d.velocidade_ms, "pegada %.1f x %.1f m" % (d.largura_pegada_m, d.comprimento_pegada_m),
          "esp=%.1f m" % d.espacamento_m(0.2), "base", area.proj.ponto_lonlat(*d.partida_xy))

planos = {}
for rot, met, kw in [("faixas", "faixas", {}), ("kmeans", "kmeans", {}), ("darp", "darp", {}),
                     ("darp+ref", "darp", {"refinar": 4, "sementes": "centros"})]:
    p = planejar_missao(area, frota, met, **kw)
    planos[rot] = p
    m = p.metricas
    print(f"{rot:9s} makespan {m['makespan_s'] / 60:.2f} min  CV {m['cv_tempo']:.3f}  dist {m['distancia_total_m'] / 1000:.2f} km  "
          f"curvas {m['curvas_total']}  cobertura {m['cobertura_pct']:.2f}%  t_alg {m['tempo_algoritmo_s']:.2f}s")

# figura comparativa 2x2
estilo()
fig, axs = plt.subplots(2, 2, figsize=(11, 8.6))
titulos = {"faixas": "(a) Faixas ponderadas", "kmeans": "(b) k-means + Voronoi",
           "darp": "(c) DARP (sementes nas bases)", "darp+ref": "(d) DARP + refinamento por tempo"}
for ax, (rot, p) in zip(axs.ravel(), planos.items()):
    m = p.metricas
    desenhar_plano(ax, area, frota, p.particao.regioes, p.rotas, None, legenda=False)
    ax.set_title(f"{titulos[rot]}\nmakespan {m['makespan_s'] / 60:.1f} min, CV {m['cv_tempo']:.2f}, "
                 f"{m['distancia_total_m'] / 1000:.1f} km, {m['curvas_total']} curvas", loc="left", fontsize=8.5)
    ax.tick_params(labelsize=7)
from matplotlib.patches import Patch  # noqa: E402
from cerise_drones.visualizar import CORES  # noqa: E402
fig.legend(handles=[Patch(facecolor=CORES[i], alpha=0.5, label=f"{d.id} - tipo {d.tipo} ({d.rotulo})") for i, d in enumerate(frota)],
           loc="lower center", ncol=4, fontsize=7.5)
fig.suptitle("Campus Samambaia/UFG (34,2 ha) - particao entre 4 drones heterogeneos", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
fig.savefig(EV / "01_ufg_comparacao_metodos.png", bbox_inches="tight")
plt.close(fig)

final = planos["darp+ref"]
figura_plano(final, EV / "02_ufg_plano_darp_ref.png",
             "Campus Samambaia/UFG - DARP + refinamento (4 drones heterogeneos, 40 m AGL)")
arqs = final.exportar(EXP, "ufg_darp_ref")
print("exportados:", [a.name for a in arqs])
html = mapa_folium(final, EXP / "mapa_ufg_darp_ref.html",
                   "Campus Samambaia/UFG - segmentacao entre 4 drones (DARP + refinamento)")
capturar_html(html, EV / "03_ufg_mapa_folium.png")
json.dump({k: p.metricas for k, p in planos.items()}, open(RES / "demo_ufg_metricas.json", "w"), indent=1, default=float)
json.dump({"frota": [d.para_dict() for d in frota], "bases_lonlat": [area.proj.ponto_lonlat(*d.partida_xy) for d in frota],
           "area_ha": area.livre.area / 1e4, "epsg": area.proj.epsg}, open(RES / "demo_ufg_frota.json", "w"), indent=1, default=float)
print("ok")
