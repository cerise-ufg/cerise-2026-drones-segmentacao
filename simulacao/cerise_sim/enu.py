"""Coordenadas locais ENU (x = leste, y = norte, z = cima, metros) em torno de uma origem lon/lat.

Usa a projecao azimutal equidistante centrada na origem (plano tangente local), cujo erro em relacao
ao ENU geodesico e desprezivel para areas de ~1 km (o terreno e tratado como plano, z = 0 no chao).
Isaac Sim (Z para cima, metros) e Gazebo (ENU) usam esse mesmo referencial diretamente.
"""
from __future__ import annotations

import numpy as np


class ENU:
    def __init__(self, lon0: float, lat0: float, epsg_utm: int | None = None):
        from pyproj import Transformer
        self.lon0, self.lat0 = float(lon0), float(lat0)
        crs = f"+proj=aeqd +lat_0={self.lat0} +lon_0={self.lon0} +datum=WGS84 +units=m +no_defs"
        self._ll2e = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self._e2ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        self.epsg_utm = epsg_utm
        if epsg_utm:
            self._utm2e = Transformer.from_crs(f"EPSG:{epsg_utm}", crs, always_xy=True)
            self._e2utm = Transformer.from_crs(crs, f"EPSG:{epsg_utm}", always_xy=True)

    def de_lonlat(self, lon, lat):
        x, y = self._ll2e.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)

    def para_lonlat(self, x, y):
        lon, lat = self._e2ll.transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(lon), np.asarray(lat)

    def de_utm(self, x, y):
        ex, ey = self._utm2e.transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(ex), np.asarray(ey)

    def para_utm(self, x, y):
        ux, uy = self._e2utm.transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(ux), np.asarray(uy)

    def geom_de_utm(self, geom):
        import shapely
        return shapely.transform(geom, lambda xx, yy: self._utm2e.transform(xx, yy), interleaved=False)

    def geom_de_lonlat(self, geom):
        import shapely
        return shapely.transform(geom, lambda xx, yy: self._ll2e.transform(xx, yy), interleaved=False)

    def para_dict(self) -> dict:
        return {"lon0": self.lon0, "lat0": self.lat0, "epsg_utm": self.epsg_utm,
                "projecao": "aeqd (plano tangente local), x=leste y=norte z=cima, metros; chao em z=0"}

    @classmethod
    def de_dict(cls, d: dict) -> "ENU":
        return cls(d["lon0"], d["lat0"], d.get("epsg_utm"))
