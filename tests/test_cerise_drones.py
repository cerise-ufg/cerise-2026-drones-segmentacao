"""Testes do pacote cerise_drones (python -m pytest -q tests)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cerise_drones import protocolo as P  # noqa: E402
from cerise_drones.agregacao import contar, pixel_para_solo  # noqa: E402
from cerise_drones.cenarios import CENARIOS, area_ufg, frota_padrao  # noqa: E402
from cerise_drones.cobertura import Navegador, planejar_regiao, simular_missao  # noqa: E402
from cerise_drones.exportar import plano_qgc, validar_plano  # noqa: E402
from cerise_drones.frota import Drone  # noqa: E402
from cerise_drones.geo import AreaMissao, epsg_utm  # noqa: E402
from cerise_drones.metricas import cobertura_raster, pegada_rota  # noqa: E402
from cerise_drones.particao import particionar  # noqa: E402
from cerise_drones.planejador import planejar_missao  # noqa: E402


def test_utm_goiania():
    assert epsg_utm(-49.265, -16.604) == 32722
    a = area_ufg()
    assert 30e4 < a.livre.area < 40e4  # ~34 ha


@pytest.mark.parametrize("metodo,kw", [("faixas", {}), ("kmeans", {}), ("darp", {}), ("darp", {"sementes": "centros"})])
@pytest.mark.parametrize("cen", ["ufg", "concavo_U"])
def test_particao_e_exata(metodo, kw, cen):
    """As subareas sao disjuntas e cobrem toda a area livre."""
    area = CENARIOS[cen]()
    frota = frota_padrao(area, 4, True)
    r = particionar(area, frota, metodo, **kw)
    soma = sum(g.area for g in r.regioes)
    uniao = shapely.union_all(r.regioes)
    assert abs(soma - area.livre.area) / area.livre.area < 2e-3
    assert abs(uniao.area - area.livre.area) / area.livre.area < 2e-3
    for i in range(4):
        for j in range(i + 1, 4):
            assert r.regioes[i].intersection(r.regioes[j]).area < 1e-3 * area.livre.area


def test_darp_balanceia_por_peso():
    area = CENARIOS["convexo"]()
    frota = frota_padrao(area, 3, True)
    r = particionar(area, frota, "darp")
    alvo = np.array(r.pesos) * area.livre.area
    assert np.max(np.abs(np.array(r.areas()) - alvo) / alvo) < 0.03


def test_boustrophedon_cobre_regiao_e_evita_zonas():
    area = CENARIOS["concavo_U"]()
    d = Drone("D1", partida_xy=(area.livre.bounds[0] - 20, area.livre.bounds[1] - 20), altitude_m=40)
    nav = Navegador(area.obstaculos, contorno=area.zonas_buf)
    rota = planejar_regiao(d, area.livre, 0.2, nav)
    cob = cobertura_raster(area.livre, [pegada_rota(rota, d)])
    assert cob["cobertura_pct"] > 99.0
    # nenhuma perna (cobertura ou transito) atravessa as zonas proibidas originais
    for k in range(len(rota.pontos) - 1):
        seg = shapely.LineString([rota.pontos[k], rota.pontos[k + 1]])
        for z in area.zonas:
            assert seg.intersection(z).length < 1.0


def test_bateria_insere_trocas():
    area = CENARIOS["convexo"]()
    d = Drone("D1", partida_xy=area.livre.centroid.coords[0], autonomia_min=6, altitude_m=40)
    rota = planejar_regiao(d, area.livre, 0.2, None)
    m = simular_missao(rota, d)
    assert m.n_voos >= 2 and not m.viavel_um_voo
    # cada voo respeita a autonomia util
    assert m.tempo_voo_s <= m.n_voos * d.autonomia_util_s + 1e-6 + m.n_voos * 120


def test_exportacao_qgc_valida(tmp_path):
    area = area_ufg()
    frota = frota_padrao(area, 2, False)
    p = planejar_missao(area, frota, "darp")
    for d, r in zip(frota, p.rotas):
        plano = plano_qgc(area, d, r)
        assert validar_plano(plano) == []
        cmds = [it["command"] for it in plano["mission"]["items"]]
        assert cmds[0] == 22 and cmds[-1] == 20 and 206 in cmds and 16 in cmds
    arqs = p.exportar(tmp_path, "t")
    assert len(arqs) == 3 * 2 + 1
    json.loads((tmp_path / "t_D1.plan").read_text())


def test_protocolo_compativel_cerise_proto():
    assert P.crc16(b"123456789") == 0x29B1  # valor de verificacao CRC-16/CCITT-FALSE
    tam = {k: len(v) for k, v in P.exemplos().items()}
    # mesmos tamanhos de quadro da frente LoRa (resultados/airtime_mensagens.csv)
    assert tam["HEARTBEAT"] == 18 and tam["POS"] == 24 and tam["AREA_ASSIGN"] == 32
    assert tam["COVERAGE_DONE"] == 17 and tam["COUNT_REPORT"] == 23 and tam["ACK"] == 16
    assert all(v <= 8 + 32 + 2 for v in tam.values())
    q = P.exemplos()["POS"]
    d = P.decodificar(q)
    assert d["tipo"] == "POS" and d["campos"][0] == P.e7(-16.604)
    with pytest.raises(ValueError):
        P.decodificar(q[:-1] + bytes([q[-1] ^ 0xFF]))
    assert abs(P.tempo_no_ar_ms(18, 7) - 51.5) < 0.1  # = airtime_mensagens.csv (SF7)


def test_area_poly_fragmentos():
    area = area_ufg()
    qs = P.area_assign_poligono(1, 1, area.area_ll, 40, 44.8, proj=area.proj)
    assert P.decodificar(qs[0])["tipo"] == "AREA_ASSIGN"
    verts = []
    for q in qs[1:]:
        d = P.decodificar(q)
        assert d["tipo"] == "AREA_POLY"
        verts += d["vertices"]
    assert len(verts) >= 4


def test_pixel_para_solo_e_contagem():
    x, y = pixel_para_solo(4000, 1500, 4000, 3000, (0.0, 0.0), 40.0, 70.0, 55.0, 0.0)
    assert abs(x - 40 * math.tan(math.radians(35))) < 1e-6 and abs(y) < 1e-6
    x, y = pixel_para_solo(2000, 0, 4000, 3000, (0.0, 0.0), 40.0, 70.0, 55.0, 90.0)  # topo -> leste
    assert abs(y) < 1e-6 and x > 0
    a = shapely.box(0, 0, 100, 100); b = shapely.box(100, 0, 200, 100)
    det = np.array([[10, 10], [10.5, 10.2], [150, 50], [99.5, 50], [100.6, 50.3]])
    dr = np.array([0, 0, 1, 0, 1])
    c = contar(det, dr, [a, b])
    assert c["soma_ingenua"] == 5
    assert c["proposto_posse_dedup_borda"] == 3  # pessoa duplicada no interior + pessoa vista pelos 2 na fronteira
