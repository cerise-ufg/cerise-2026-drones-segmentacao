"""Conversao geografica (WGS84 lon/lat <-> UTM metrico) e carga da area de missao.

Todo o planejamento e feito em coordenadas metricas UTM (zona escolhida pelo centroide da area,
datum WGS84: EPSG 327xx no hemisferio sul). As exportacoes (QGC .plan, KML, GeoJSON, folium)
voltam para lon/lat.
"""
from __future__ import annotations

import json
from pathlib import Path

import shapely
import shapely.affinity
from pyproj import Transformer
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union


def epsg_utm(lon: float, lat: float) -> int:
    """EPSG da zona UTM/WGS84 que contem o ponto (ex.: Goiania -> 32722, UTM 22S)."""
    zona = int((lon + 180.0) // 6.0) + 1
    return (32700 if lat < 0 else 32600) + zona


class Projetor:
    """Transforma geometrias/pontos entre lon/lat (EPSG:4326) e UTM (metros)."""

    def __init__(self, lon0: float, lat0: float, epsg: int | None = None):
        self.epsg = epsg or epsg_utm(lon0, lat0)
        self._fwd = Transformer.from_crs(4326, self.epsg, always_xy=True)
        self._inv = Transformer.from_crs(self.epsg, 4326, always_xy=True)

    def xy(self, geom):
        return shapely.transform(geom, lambda x, y: self._fwd.transform(x, y), interleaved=False)

    def lonlat(self, geom):
        return shapely.transform(geom, lambda x, y: self._inv.transform(x, y), interleaved=False)

    def ponto_xy(self, lon: float, lat: float) -> tuple[float, float]:
        x, y = self._fwd.transform(lon, lat)
        return float(x), float(y)

    def ponto_lonlat(self, x: float, y: float) -> tuple[float, float]:
        lon, lat = self._inv.transform(x, y)
        return float(lon), float(lat)


def ler_geometrias(fonte) -> list:
    """Le geometrias de um arquivo GeoJSON (FeatureCollection/Feature/Geometry) ou de um dict."""
    if isinstance(fonte, (str, Path)):
        with open(fonte, encoding="utf-8") as f:
            fonte = json.load(f)
    if fonte.get("type") == "FeatureCollection":
        return [shape(ft["geometry"]) for ft in fonte["features"]]
    if fonte.get("type") == "Feature":
        return [shape(fonte["geometry"])]
    return [shape(fonte)]


def poligonos(geom) -> list[Polygon]:
    """Lista os poligonos (nao vazios) de uma geometria qualquer."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        out = []
        for g in geom.geoms:
            out.extend(poligonos(g))
        return out
    return []


def so_poligonos(geom):
    """Remove linhas/pontos residuais de operacoes booleanas, devolvendo (Multi)Polygon."""
    ps = [p for p in poligonos(geom) if p.area > 1e-6]
    if not ps:
        return Polygon()
    return ps[0] if len(ps) == 1 else MultiPolygon(ps)


class AreaMissao:
    """Area de atuacao (poligono) menos as zonas proibidas (com margem de seguranca), em metros."""

    def __init__(self, nome: str, area_ll, zonas_ll=(), margem_zona_m: float = 10.0,
                 projetor: Projetor | None = None):
        self.nome = nome
        self.area_ll = shapely.make_valid(area_ll)
        self.zonas_ll = [shapely.make_valid(z) for z in zonas_ll]
        c = self.area_ll.centroid
        self.proj = projetor or Projetor(c.x, c.y)
        self.margem_zona_m = margem_zona_m
        self.area = so_poligonos(shapely.make_valid(self.proj.xy(self.area_ll)))
        self.zonas = [so_poligonos(self.proj.xy(z)) for z in self.zonas_ll]
        if self.zonas:
            self.zonas_buf = unary_union([z.buffer(margem_zona_m, quad_segs=8) for z in self.zonas])
        else:
            self.zonas_buf = Polygon()
        self.livre = so_poligonos(self.area.difference(self.zonas_buf))
        # obstaculos para o transito (um pouco menores que a zona com margem, para que os
        # vertices do contorno com margem sejam pontos validos do grafo de visibilidade)
        self.obstaculos = self.zonas_buf.buffer(-0.05 * margem_zona_m) if self.zonas else Polygon()

    @classmethod
    def de_arquivos(cls, nome, arquivo_area, arquivos_zonas=(), margem_zona_m=10.0):
        area = unary_union(ler_geometrias(arquivo_area))
        zonas = []
        for z in arquivos_zonas:
            zonas.extend(ler_geometrias(z))
        return cls(nome, area, zonas, margem_zona_m)

    @classmethod
    def de_metros(cls, nome, area_local, zonas_local=(), lon0=-49.2650, lat0=-16.6040,
                  margem_zona_m=10.0):
        """Cria uma area a partir de poligonos em metros locais (origem em lon0/lat0).

        Usado nos cenarios sinteticos: as coordenadas locais sao deslocadas para a UTM do
        ponto de referencia, de modo que as exportacoes geograficas continuem validas."""
        proj = Projetor(lon0, lat0)
        x0, y0 = proj.ponto_xy(lon0, lat0)
        desloca = lambda g: shapely.affinity.translate(g, x0, y0)  # noqa: E731
        area_ll = proj.lonlat(desloca(area_local))
        zonas_ll = [proj.lonlat(desloca(z)) for z in zonas_local]
        obj = cls(nome, area_ll, zonas_ll, margem_zona_m, projetor=proj)
        obj.origem_local = (x0, y0)
        return obj

    def geojson_ll(self) -> dict:
        feats = [{"type": "Feature", "properties": {"papel": "area", "nome": self.nome},
                  "geometry": mapping(self.area_ll)}]
        for i, z in enumerate(self.zonas_ll):
            feats.append({"type": "Feature", "properties": {"papel": "zona_proibida", "indice": i},
                          "geometry": mapping(z)})
        return {"type": "FeatureCollection", "features": feats}

