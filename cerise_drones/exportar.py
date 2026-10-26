"""Exportacao por drone: QGroundControl .plan (itens MAVLink), KML e GeoJSON."""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from shapely.geometry import LineString, Point, mapping

from .cobertura import Rota
from .frota import Drone
from .geo import AreaMissao, poligonos

# Comandos MAVLink usados
MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_DO_CHANGE_SPEED = 178
MAV_CMD_DO_SET_CAM_TRIGG_DIST = 206
MAV_FRAME_MISSION = 2
MAV_FRAME_GLOBAL_RELATIVE_ALT = 3

CORES_KML = ["ffd6782a", "ff3468eb", "ff7aaf1b", "ff00a1ed", "ffa47be8", "ff008300", "ffd65062", "ff4849e3"]


def _item(seq, cmd, frame, params, alt=None):
    it = {"autoContinue": True, "command": cmd, "doJumpId": seq, "frame": frame,
          "params": params, "type": "SimpleItem"}
    if alt is not None:
        it.update({"AMSLAltAboveTerrain": None, "Altitude": alt, "AltitudeMode": 1})
    return it


def plano_qgc(area: AreaMissao, drone: Drone, rota: Rota, sobreposicao_frontal: float = 0.7,
              firmware: int = 12) -> dict:
    """Plano QGroundControl (fileType Plan, versao 1). firmware 12 = PX4, 3 = ArduPilot.

    Sequencia: TAKEOFF -> DO_CHANGE_SPEED -> waypoints; DO_SET_CAM_TRIGG_DIST liga a camera no
    inicio de cada trecho de cobertura e desliga no fim -> RTL. A geocerca inclui a area de missao
    (inclusao) e as zonas proibidas (exclusao)."""
    proj = area.proj
    alt = drone.altitude_m
    hx, hy = drone.partida_xy
    hlon, hlat = proj.ponto_lonlat(hx, hy)
    seq = 1
    itens = [_item(seq, MAV_CMD_NAV_TAKEOFF, MAV_FRAME_GLOBAL_RELATIVE_ALT, [0, 0, 0, None, hlat, hlon, alt], alt)]
    seq += 1
    itens.append(_item(seq, MAV_CMD_DO_CHANGE_SPEED, MAV_FRAME_MISSION, [1, drone.velocidade_ms, -1, 0, 0, 0, 0]))
    seq += 1
    dist_foto = round(drone.distancia_disparo_m(sobreposicao_frontal), 2)
    camera_on = False
    P = rota.pontos
    for k in range(1, len(P) - 1):  # o primeiro e o ultimo ponto sao a base
        cobre_saindo = bool(rota.cobre[k]) if k < len(rota.cobre) else False
        lon, lat = proj.ponto_lonlat(*P[k])
        itens.append(_item(seq, MAV_CMD_NAV_WAYPOINT, MAV_FRAME_GLOBAL_RELATIVE_ALT,
                           [0, 0, 0, None, round(lat, 8), round(lon, 8), alt], alt))
        seq += 1
        if cobre_saindo and not camera_on:
            itens.append(_item(seq, MAV_CMD_DO_SET_CAM_TRIGG_DIST, MAV_FRAME_MISSION, [dist_foto, 0, 1, 0, 0, 0, 0]))
            seq += 1
            camera_on = True
        elif not cobre_saindo and camera_on:
            itens.append(_item(seq, MAV_CMD_DO_SET_CAM_TRIGG_DIST, MAV_FRAME_MISSION, [0, 0, 1, 0, 0, 0, 0]))
            seq += 1
            camera_on = False
    if camera_on:
        itens.append(_item(seq, MAV_CMD_DO_SET_CAM_TRIGG_DIST, MAV_FRAME_MISSION, [0, 0, 1, 0, 0, 0, 0]))
        seq += 1
    itens.append(_item(seq, MAV_CMD_NAV_RETURN_TO_LAUNCH, MAV_FRAME_MISSION, [0, 0, 0, 0, 0, 0, 0]))

    def anel(p):
        return [[round(lat, 8), round(lon, 8)] for lon, lat in list(p.exterior.coords)[:-1]]

    cercas = [{"inclusion": True, "polygon": anel(p), "version": 1}
              for p in poligonos(proj.lonlat(area.area.buffer(30.0)))]
    for z in area.zonas_ll:
        cercas += [{"inclusion": False, "polygon": anel(p), "version": 1} for p in poligonos(z)]
    return {
        "fileType": "Plan",
        "geoFence": {"circles": [], "polygons": cercas, "version": 2},
        "groundStation": "QGroundControl",
        "mission": {"cruiseSpeed": drone.velocidade_ms, "firmwareType": firmware, "globalPlanAltitudeMode": 1,
                    "hoverSpeed": 5, "items": itens, "plannedHomePosition": [round(hlat, 8), round(hlon, 8), 0],
                    "vehicleType": 2, "version": 2},
        "rallyPoints": {"points": [], "version": 2},
        "version": 1,
    }


def validar_plano(plano: dict) -> list[str]:
    """Checagem estrutural do .plan (campos exigidos pelo QGroundControl)."""
    erros = []
    if plano.get("fileType") != "Plan":
        erros.append("fileType")
    m = plano.get("mission", {})
    for k in ("items", "plannedHomePosition", "firmwareType", "vehicleType", "version"):
        if k not in m:
            erros.append(f"mission.{k}")
    for i, it in enumerate(m.get("items", [])):
        if it.get("type") != "SimpleItem" or len(it.get("params", [])) != 7:
            erros.append(f"item {i}")
        if it.get("doJumpId") != i + 1:
            erros.append(f"doJumpId {i}")
        if it["command"] == MAV_CMD_NAV_WAYPOINT:
            lat, lon = it["params"][4], it["params"][5]
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                erros.append(f"coord {i}")
    return erros


def geojson_drone(area: AreaMissao, drone: Drone, rota: Rota, regiao) -> dict:
    proj = area.proj
    feats = []
    if regiao is not None and not regiao.is_empty:
        feats.append({"type": "Feature", "properties": {"drone": drone.id, "papel": "subarea",
                                                        "area_m2": round(regiao.area, 1)},
                      "geometry": mapping(proj.lonlat(regiao))})
    ll = [proj.ponto_lonlat(*p) for p in rota.pontos]
    feats.append({"type": "Feature", "properties": {"drone": drone.id, "papel": "rota", "altitude_m": drone.altitude_m,
                                                    "distancia_m": round(rota.distancia_m, 1),
                                                    "angulo_varredura_graus": round(rota.angulo, 2)},
                  "geometry": {"type": "LineString", "coordinates": [[round(x, 8), round(y, 8), drone.altitude_m] for x, y in ll]}})
    for k, (x, y) in enumerate(ll):
        feats.append({"type": "Feature", "properties": {"drone": drone.id, "papel": "waypoint", "seq": k,
                                                        "cobertura_saindo": bool(rota.cobre[k]) if k < len(rota.cobre) else False},
                      "geometry": {"type": "Point", "coordinates": [round(x, 8), round(y, 8)]}})
    return {"type": "FeatureCollection", "features": feats}


def kml(area: AreaMissao, frota: list[Drone], rotas: list[Rota], regioes: list, nome: str) -> str:
    proj = area.proj
    out = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2">',
           f"<Document><name>{escape(nome)}</name>"]
    for i, _ in enumerate(frota):
        c = CORES_KML[i % len(CORES_KML)]
        out.append(f'<Style id="d{i}"><LineStyle><color>{c}</color><width>2</width></LineStyle>'
                   f'<PolyStyle><color>40{c[2:]}</color></PolyStyle></Style>')
    out.append('<Style id="zona"><LineStyle><color>ff0000ff</color><width>2</width></LineStyle>'
               '<PolyStyle><color>600000ff</color></PolyStyle></Style>')

    def coords_poly(p):
        return " ".join(f"{x:.8f},{y:.8f},0" for x, y in p.exterior.coords)

    for z in area.zonas_ll:
        for p in poligonos(z):
            out.append(f'<Placemark><name>Zona proibida</name><styleUrl>#zona</styleUrl><Polygon><outerBoundaryIs>'
                       f'<LinearRing><coordinates>{coords_poly(p)}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>')
    for i, (d, r, g) in enumerate(zip(frota, rotas, regioes)):
        out.append(f"<Folder><name>{d.id}</name>")
        if g is not None:
            for p in poligonos(proj.lonlat(g)):
                out.append(f'<Placemark><name>{d.id} subarea</name><styleUrl>#d{i}</styleUrl><Polygon><outerBoundaryIs>'
                           f'<LinearRing><coordinates>{coords_poly(p)}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>')
        ll = " ".join(f"{x:.8f},{y:.8f},{d.altitude_m:.1f}" for x, y in (proj.ponto_lonlat(*p) for p in r.pontos))
        out.append(f'<Placemark><name>{d.id} rota</name><styleUrl>#d{i}</styleUrl><LineString><altitudeMode>relativeToGround</altitudeMode>'
                   f'<coordinates>{ll}</coordinates></LineString></Placemark>')
        hx, hy = proj.ponto_lonlat(*d.partida_xy)
        out.append(f"<Placemark><name>{d.id} base</name><Point><coordinates>{hx:.8f},{hy:.8f},0</coordinates></Point></Placemark>")
        out.append("</Folder>")
    out.append("</Document></kml>")
    return "\n".join(out)


def exportar_tudo(pasta, area: AreaMissao, frota, rotas, regioes, sobreposicao_frontal=0.7, nome="missao"):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    arquivos = []
    for d, r, g in zip(frota, rotas, regioes):
        plano = plano_qgc(area, d, r, sobreposicao_frontal)
        erros = validar_plano(plano)
        if erros:
            raise ValueError(f"plano invalido {d.id}: {erros}")
        f = pasta / f"{nome}_{d.id}.plan"
        f.write_text(json.dumps(plano, indent=1), encoding="utf-8")
        arquivos.append(f)
        f = pasta / f"{nome}_{d.id}.geojson"
        f.write_text(json.dumps(geojson_drone(area, d, r, g)), encoding="utf-8")
        arquivos.append(f)
        f = pasta / f"{nome}_{d.id}.kml"
        f.write_text(kml(area, [d], [r], [g], f"{nome} {d.id}"), encoding="utf-8")
        arquivos.append(f)
    f = pasta / f"{nome}_todos.kml"
    f.write_text(kml(area, frota, rotas, regioes, nome), encoding="utf-8")
    arquivos.append(f)
    return arquivos
