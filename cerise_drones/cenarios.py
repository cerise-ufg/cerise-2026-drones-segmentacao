"""Cenarios de benchmark: area real do Campus Samambaia (UFG) + poligonos sinteticos."""
from __future__ import annotations

from pathlib import Path

from shapely.geometry import Point, Polygon, box

from .frota import criar_frota
from .geo import AreaMissao, poligonos

DADOS = Path(__file__).resolve().parent.parent / "dados"

# Referencia para os cenarios sinteticos (centro aproximado do Campus Samambaia, Goiania-GO)
LON0, LAT0 = -49.2650, -16.6040


def area_ufg(com_zona: bool = False) -> AreaMissao:
    """Campo aberto (landuse=meadow, OSM way 362871566) no setor oeste do Campus Samambaia/UFG,
    ~34 ha, poligono concavo. A zona proibida opcional e HIPOTETICA (circulo de 45 m de raio,
    representando, p.ex., uma estrutura ou area de pouso a evitar)."""
    zonas = []
    if com_zona:
        from shapely.geometry import shape
        zonas = [Point(-49.2735, -16.6040).buffer(0.00042, quad_segs=16)]  # ~45 m em lat/lon
    a = AreaMissao.de_arquivos("UFG Samambaia - campo aberto", DADOS / "ufg_samambaia_campo_aberto.geojson",
                               margem_zona_m=10.0)
    if zonas:
        a = AreaMissao(a.nome + " + zona", a.area_ll, zonas, 10.0)
    return a


def area_convexa() -> AreaMissao:
    p = Polygon([(0, 0), (1000, -80), (1250, 420), (1080, 780), (250, 830), (-80, 450)])
    return AreaMissao.de_metros("Sintetico convexo", p, [], LON0 + 0.02, LAT0 - 0.02)


def area_concava_L() -> AreaMissao:
    p = Polygon([(0, 0), (900, 0), (900, 300), (380, 300), (380, 750), (0, 750)])
    zona = Point(620, 150).buffer(45, quad_segs=16)
    return AreaMissao.de_metros("Sintetico concavo (L) + 1 zona", p, [zona], LON0 + 0.02, LAT0 + 0.02)


def area_concava_U() -> AreaMissao:
    p = Polygon([(0, 0), (950, 0), (950, 700), (650, 700), (650, 280), (300, 280), (300, 700), (0, 700)])
    zonas = [box(120, 420, 200, 520), Point(800, 140).buffer(40, quad_segs=16)]
    return AreaMissao.de_metros("Sintetico concavo (U) + 2 zonas", p, zonas, LON0 - 0.02, LAT0 + 0.02)


CENARIOS = {
    "ufg": lambda: area_ufg(False),
    "convexo": area_convexa,
    "concavo_L": area_concava_L,
    "concavo_U": area_concava_U,
}


def bases_no_contorno(area: AreaMissao, n: int, deslocamento: float = 0.07, recuo_m: float = 15.0):
    """n bases igualmente espacadas ao longo do contorno externo (recuadas recuo_m para fora)."""
    ext = poligonos(area.area)[0].exterior
    out = []
    for k in range(n):
        p = ext.interpolate((deslocamento + k / n) % 1.0, normalized=True)
        c = area.area.centroid
        dx, dy = p.x - c.x, p.y - c.y
        L = max(1e-9, (dx * dx + dy * dy) ** 0.5)
        out.append((p.x + recuo_m * dx / L, p.y + recuo_m * dy / L))
    return out


def frota_padrao(area: AreaMissao, n: int, heterogenea: bool, altitude_m: float = 40.0):
    tipos = (["A", "B", "C"] * 3)[:n] if heterogenea else ["A"] * n
    return criar_frota(tipos, bases_no_contorno(area, n), altitude_m=altitude_m)
