"""Visualizacao: figura matplotlib (PNG) do plano e mapa interativo folium (HTML)."""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Polygon as MplPolygon  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .geo import poligonos  # noqa: E402

# paleta categorica validada (ordem fixa; ver skill de visualizacao): D1, D2, ...
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TINTA = "#0b0b0b"
TINTA2 = "#52514e"
GRADE = "#e4e3df"
SUPERFICIE = "#fcfcfb"


def estilo():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10,
        "axes.edgecolor": "#b5b4ae", "axes.labelcolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA2,
        "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": SUPERFICIE,
        "legend.frameon": False,
    })


def _desenha_geom(ax, geom, ox, oy, **kw):
    for p in poligonos(geom):
        xy = [(x - ox, y - oy) for x, y in p.exterior.coords]
        ax.add_patch(MplPolygon(xy, closed=True, **kw))
        for h in p.interiors:
            ax.add_patch(MplPolygon([(x - ox, y - oy) for x, y in h.coords], closed=True,
                                    facecolor="white", edgecolor=kw.get("edgecolor", "none"), linewidth=0.5))


def desenhar_plano(ax, area, frota, regioes, rotas=None, titulo=None, mostrar_rotas=True, legenda=True,
                   rotulos=True, origem=None):
    ox, oy = origem if origem is not None else (area.area.bounds[0], area.area.bounds[1])
    _desenha_geom(ax, area.area, ox, oy, facecolor="#f0efec", edgecolor="#8a8984", linewidth=0.8)
    for i, (d, g) in enumerate(zip(frota, regioes)):
        if g is None or g.is_empty:
            continue
        _desenha_geom(ax, g, ox, oy, facecolor=CORES[i % len(CORES)], alpha=0.22,
                      edgecolor=CORES[i % len(CORES)], linewidth=1.0)
    for z in area.zonas:
        _desenha_geom(ax, z, ox, oy, facecolor="#d03b3b", alpha=0.35, edgecolor="#d03b3b", linewidth=0.8,
                      hatch="////")
    if not area.zonas_buf.is_empty:
        for p in poligonos(area.zonas_buf):
            xs, ys = zip(*[(x - ox, y - oy) for x, y in p.exterior.coords])
            ax.plot(xs, ys, color="#d03b3b", lw=0.7, ls="--")
    if rotas is not None and mostrar_rotas:
        for i, (d, r) in enumerate(zip(frota, rotas)):
            P = r.pontos
            c = CORES[i % len(CORES)]
            for k in range(len(P) - 1):
                ax.plot([P[k, 0] - ox, P[k + 1, 0] - ox], [P[k, 1] - oy, P[k + 1, 1] - oy], color=c,
                        lw=1.1 if r.cobre[k] else 0.8, ls="-" if r.cobre[k] else ":", solid_capstyle="round")
    for i, d in enumerate(frota):
        bx, by = d.partida_xy
        ax.plot(bx - ox, by - oy, marker="^", ms=8, color=CORES[i % len(CORES)], mec="white", mew=1.2, zorder=5)
        if rotulos:
            g = regioes[i]
            if g is not None and not g.is_empty:
                rp = g.representative_point()
                ax.text(rp.x - ox, rp.y - oy, d.id, ha="center", va="center", fontsize=8, color=TINTA,
                        fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8), zorder=6)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("Leste (m)")
    ax.set_ylabel("Norte (m)")
    if titulo:
        ax.set_title(titulo, loc="left", color=TINTA)
    if legenda:
        h = [Patch(facecolor=CORES[i % len(CORES)], alpha=0.5, edgecolor=CORES[i % len(CORES)],
                   label=f"{d.id} (tipo {d.tipo})") for i, d in enumerate(frota)]
        h.append(Line2D([], [], marker="^", color=TINTA2, ls="none", label="base de decolagem"))
        if area.zonas:
            h.append(Patch(facecolor="#d03b3b", alpha=0.35, hatch="////", label="zona proibida"))
        if rotas is not None and mostrar_rotas:
            h.append(Line2D([], [], color=TINTA2, lw=1.1, label="varredura (camera ligada)"))
            h.append(Line2D([], [], color=TINTA2, lw=0.8, ls=":", label="transito / curva"))
        ax.legend(handles=h, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5)


def figura_plano(plano, caminho, titulo=None):
    estilo()
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    m = plano.metricas
    t = titulo or f"{plano.area.nome} - {m['metodo']} - {len(plano.frota)} drones"
    desenhar_plano(ax, plano.area, plano.frota, plano.particao.regioes, plano.rotas, t)
    txt = (f"makespan {m['makespan_s'] / 60:.1f} min | CV tempo {m['cv_tempo']:.3f} | "
           f"dist. total {m['distancia_total_m'] / 1000:.2f} km | curvas {m['curvas_total']} | "
           f"cobertura {m['cobertura_pct']:.2f}%")
    fig.text(0.01, 0.01, txt, fontsize=7.5, color=TINTA2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)


def mapa_folium(plano, caminho_html, titulo=None):
    """Mapa interativo com imagem de satelite (Esri World Imagery) + OSM, uma camada por drone."""
    import folium

    area = plano.area
    proj = area.proj
    c = area.area_ll.centroid
    mapa = folium.Map(location=[c.y, c.x], zoom_start=16, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=False, overlay=False).add_to(mapa)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Imagem: Esri, Maxar, Earthstar Geographics", name="Satelite (Esri)", max_zoom=19,
                     show=True, overlay=False).add_to(mapa)
    folium.GeoJson(area.geojson_ll()["features"][0], name="Area de atuacao",
                   style_function=lambda f: {"color": "#ffffff", "weight": 2.5, "fillOpacity": 0.0, "dashArray": "6 4"}).add_to(mapa)
    for z in area.zonas_ll:
        folium.GeoJson(z.__geo_interface__, name="Zona proibida",
                       style_function=lambda f: {"color": "#d03b3b", "weight": 2, "fillColor": "#d03b3b", "fillOpacity": 0.45}).add_to(mapa)
    for i, (d, r, g) in enumerate(zip(plano.frota, plano.rotas, plano.particao.regioes)):
        cor = CORES[i % len(CORES)]
        fg = folium.FeatureGroup(name=f"{d.id} (tipo {d.tipo})")
        if g is not None and not g.is_empty:
            folium.GeoJson(proj.lonlat(g).__geo_interface__,
                           style_function=lambda f, cor=cor: {"color": cor, "weight": 2, "fillColor": cor, "fillOpacity": 0.25},
                           tooltip=f"{d.id}: {g.area / 1e4:.2f} ha").add_to(fg)
        ll = [proj.ponto_lonlat(*p) for p in r.pontos]
        folium.PolyLine([(la, lo) for lo, la in ll], color=cor, weight=2.5, opacity=0.95,
                        tooltip=f"{d.id}: {r.distancia_m / 1000:.2f} km").add_to(fg)
        hx, hy = proj.ponto_lonlat(*d.partida_xy)
        folium.CircleMarker((hy, hx), radius=7, color="white", weight=2, fill=True, fill_color=cor, fill_opacity=1.0,
                            tooltip=f"Base {d.id}").add_to(fg)
        fg.add_to(mapa)
    m = plano.metricas
    t = titulo or f"{area.nome} - {m['metodo']} - {len(plano.frota)} drones"
    linhas = "".join(
        f"<div><span style='display:inline-block;width:10px;height:10px;background:{CORES[i % len(CORES)]};"
        f"margin-right:6px;border-radius:2px'></span>{d.id} tipo {d.tipo}: {pd['area_m2'] / 1e4:.1f} ha, "
        f"{pd['tempo_s'] / 60:.1f} min</div>"
        for i, (d, pd) in enumerate(zip(plano.frota, m["por_drone"])))
    html = (f"<div style='position:fixed;top:10px;left:50px;z-index:9999;background:rgba(255,255,255,.92);"
            f"padding:8px 12px;border-radius:6px;font:12px/1.4 sans-serif;color:#0b0b0b;box-shadow:0 1px 4px rgba(0,0,0,.3)'>"
            f"<b>{t}</b><br>makespan {m['makespan_s'] / 60:.1f} min &middot; cobertura {m['cobertura_pct']:.1f}%{linhas}</div>")
    mapa.get_root().html.add_child(folium.Element(html))
    folium.LayerControl(collapsed=False).add_to(mapa)
    b = area.area_ll.bounds
    mapa.fit_bounds([[b[1], b[0]], [b[3], b[2]]], padding=(70, 70))
    mapa.save(str(caminho_html))
    return caminho_html


def capturar_html(caminho_html, caminho_png, largura=1400, altura=1000, espera_ms=5000):
    """Captura o mapa HTML com Playwright (Chromium headless)."""
    from pathlib import Path
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        nav = p.chromium.launch()
        pg = nav.new_page(viewport={"width": largura, "height": altura})
        pg.goto(Path(caminho_html).resolve().as_uri(), wait_until="networkidle", timeout=90_000)
        pg.wait_for_timeout(espera_ms)
        pg.screenshot(path=str(caminho_png))
        nav.close()
    return caminho_png
