"""Layout procedural do terreno em ENU (independente do simulador).

Nao usa imagens de satelite nem pegadas reais de edificios: o gramado e o poligono da area aberta do
Campus Samambaia (OSM, ODbL, ja usado pela frente de drones); caminhos, rua perimetral, edificios
esquematicos e arvores sao gerados proceduralmente (deterministicos pela semente) e NAO correspondem
as construcoes reais do campus. Tudo e exportado como malhas trianguladas (vertices/faces + UV
planar) para o Isaac Sim (UsdGeom.Mesh) e o Gazebo (OBJ), sem depender de shapely nos simuladores.
"""
from __future__ import annotations

import math

import numpy as np
import shapely
import shapely.affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from .texturas import CATALOGO


def _poligonos(g):
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    out = []
    for x in getattr(g, "geoms", []):
        out += _poligonos(x)
    return out


def triangular(geom, z: float = 0.0, metros_por_ladrilho: float = 8.0) -> dict:
    """Triangulacao de Delaunay restrita (shapely >= 2.1) de (Multi)Polygon -> malha."""
    V, F = [], []
    idx = {}
    for p in _poligonos(geom):
        p = shapely.set_precision(p, 0.01)
        tris = shapely.constrained_delaunay_triangles(p)
        for t in getattr(tris, "geoms", []):
            cs = list(t.exterior.coords)[:3]
            # orientacao anti-horaria (normal +z)
            (x1, y1), (x2, y2), (x3, y3) = cs
            if (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) < 0:
                cs = [cs[0], cs[2], cs[1]]
            face = []
            for (x, y) in cs:
                k = (round(x, 2), round(y, 2))
                if k not in idx:
                    idx[k] = len(V)
                    V.append([k[0], k[1], z])
                face.append(idx[k])
            F.append(face)
    V = np.array(V, float).reshape(-1, 3)
    uv = V[:, :2] / metros_por_ladrilho
    return {"vertices": V.round(3).tolist(), "faces": F, "uv": uv.round(4).tolist()}


def caixa_malha(pegada: Polygon, altura: float) -> dict:
    """Prisma (paredes + telhado) de uma pegada retangular; UV em metros/ladrilho por face."""
    cs = list(pegada.exterior.coords)[:-1]
    if Polygon(cs).exterior.is_ccw is False:
        cs = cs[::-1]
    V, F, UV = [], [], []
    per = 0.0
    for i in range(len(cs)):
        (x1, y1), (x2, y2) = cs[i], cs[(i + 1) % len(cs)]
        L = math.hypot(x2 - x1, y2 - y1)
        b = len(V)
        V += [[x1, y1, 0], [x2, y2, 0], [x2, y2, altura], [x1, y1, altura]]
        UV += [[per, 0], [per + L, 0], [per + L, altura], [per, altura]]
        F += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
        per += L
    b = len(V)
    for (x, y) in cs:
        V.append([x, y, altura]); UV.append([x, y])
    for i in range(1, len(cs) - 1):
        F.append([b, b + i, b + i + 1])
    return {"vertices": np.round(V, 3).tolist(), "faces": F, "uv_metros": np.round(UV, 3).tolist(),
            "faces_parede": 2 * len(cs)}


def gerar_layout(area: Polygon, seed: int = 0, margem_m: float = 140.0, n_caminhos: int = 3,
                 n_edificios: int = 14, n_arvores_dentro: int = 18, n_arvores_fora: int = 60) -> dict:
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = area.bounds
    chao = box(minx - margem_m, miny - margem_m, maxx + margem_m, maxy + margem_m)
    # rua perimetral (7 m de asfalto a 11-18 m da borda do gramado) e calcada
    rua = area.buffer(18.0, join_style="round").difference(area.buffer(11.0, join_style="round")).intersection(chao)
    calcada = area.buffer(11.0).difference(area.buffer(8.5)).intersection(chao)
    # caminhos de terra/concreto atravessando o gramado entre pontos opostos do contorno
    caminhos = []
    ext = area.exterior
    for k in range(n_caminhos):
        for _ in range(200):
            s0 = rng.uniform(0, 1)
            s1 = (s0 + rng.uniform(0.35, 0.65)) % 1.0
            a = ext.interpolate(s0, normalized=True)
            b = ext.interpolate(s1, normalized=True)
            # ponto intermediario deslocado (caminho levemente curvo)
            m = np.array([(a.x + b.x) / 2, (a.y + b.y) / 2])
            d = np.array([b.x - a.x, b.y - a.y])
            L = np.linalg.norm(d)
            if L < 120:
                continue
            n = np.array([-d[1], d[0]]) / L
            m = m + n * rng.uniform(-0.15, 0.15) * L
            linha = LineString([(a.x, a.y), tuple(m), (b.x, b.y)])
            dentro = linha.intersection(area.buffer(-0.5))
            if dentro.length > 0.8 * linha.length:
                largura = float(rng.choice([2.0, 2.5, 3.0]))
                caminhos.append({"linha": [list(map(float, c)) for c in linha.coords], "largura_m": largura,
                                 "material": "terra" if k % 2 == 0 else "concreto",
                                 "poligono": linha.buffer(largura / 2, cap_style="flat").intersection(area.buffer(9.0))})
                break
    # edificios esquematicos fora do gramado e fora da rua
    proibido = unary_union([area.buffer(26.0)])
    zona_edif = chao.buffer(-10).difference(proibido)
    edificios = []
    ocupado = []
    tent = 0
    while len(edificios) < n_edificios and tent < 5000:
        tent += 1
        x, y = rng.uniform(minx - margem_m, maxx + margem_m), rng.uniform(miny - margem_m, maxy + margem_m)
        w, h = rng.uniform(18, 60), rng.uniform(12, 30)
        ang = rng.uniform(0, 180)
        p = shapely.affinity.rotate(box(x - w / 2, y - h / 2, x + w / 2, y + h / 2), ang)
        if not zona_edif.contains(p):
            continue
        if any(p.distance(q) < 8 for q in ocupado):
            continue
        ocupado.append(p)
        andares = int(rng.integers(1, 5))
        edificios.append({"pegada": [list(map(float, c)) for c in p.exterior.coords][:-1],
                          "altura_m": round(3.4 * andares + 0.6, 2), "andares": andares})
    # arvores (tronco + copa); poucas dentro do gramado (oclusao realista), mais fora
    arvores = []
    pts_dentro = 0
    tent = 0
    while pts_dentro < n_arvores_dentro and tent < 5000:
        tent += 1
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        q = Point(x, y)
        if area.buffer(-4).contains(q) and all(q.distance(Point(t["x"], t["y"])) > 12 for t in arvores) \
                and all(q.distance(c["poligono"]) > 3 for c in caminhos):
            arvores.append({"x": x, "y": y, "altura_m": float(rng.uniform(5, 10)), "raio_copa_m": float(rng.uniform(2, 4))})
            pts_dentro += 1
    tent = 0
    n_fora = 0
    while n_fora < n_arvores_fora and tent < 10000:
        tent += 1
        x, y = rng.uniform(minx - margem_m, maxx + margem_m), rng.uniform(miny - margem_m, maxy + margem_m)
        q = Point(x, y)
        if area.buffer(1).contains(q) or rua.buffer(2).contains(q) or not chao.buffer(-3).contains(q):
            continue
        if any(q.distance(p) < 4 for p in ocupado):
            continue
        arvores.append({"x": x, "y": y, "altura_m": float(rng.uniform(5, 12)), "raio_copa_m": float(rng.uniform(2, 5))})
        n_fora += 1
    for a in arvores:
        a.update({k: round(v, 3) for k, v in a.items()})

    malhas = []
    mpl = {k: v[1] for k, v in CATALOGO.items()}
    malhas.append({"nome": "chao_externo", "material": "grama_seca", **triangular(chao.difference(area), -0.03, mpl["grama_seca"])})
    malhas.append({"nome": "gramado", "material": "grama", **triangular(area, 0.0, mpl["grama"])})
    malhas.append({"nome": "rua", "material": "asfalto", **triangular(rua, 0.01, mpl["asfalto"])})
    malhas.append({"nome": "calcada", "material": "concreto", **triangular(calcada, 0.012, mpl["concreto"])})
    for i, c in enumerate(caminhos):
        malhas.append({"nome": f"caminho_{i}", "material": c["material"],
                       **triangular(c["poligono"], 0.02, mpl[c["material"]])})
    for i, e in enumerate(edificios):
        m = caixa_malha(Polygon(e["pegada"]), e["altura_m"])
        e["malha"] = m
    for c in caminhos:
        c.pop("poligono")
    return {"chao_retangulo": [float(v) for v in chao.bounds], "caminhos": caminhos, "edificios": edificios,
            "arvores": arvores, "malhas": malhas,
            "nota": "layout procedural: caminhos, rua, edificios e arvores NAO correspondem as construcoes reais"}
