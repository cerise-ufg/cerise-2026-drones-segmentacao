"""Testes do cerise_sim com dados sinteticos (sem simulador, sem GPU).

Rodar (Windows): ..\\.venv\\Scripts\\python -m pytest -q tests
Rodar (Spark):   bash scripts/rodar_testes_spark.sh
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon, box

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM.parent))

from cerise_sim.avaliacao import avaliar_contagem, casar, pr_quadros  # noqa: E402
from cerise_sim.camera import (Intrinsecos, matriz_de_quat, olhar_para, pegada_no_chao, pixel_para_chao,  # noqa: E402
                               pose_drone, projetar, quat_de_matriz, r_usd_de_cv)
from cerise_sim.contagem import DetectorOraculo, ParamsContagem, contar_execucao  # noqa: E402
from cerise_sim.dados import Execucao  # noqa: E402
from cerise_sim.pessoas import Pessoa, gabarito, gerar_pessoas  # noqa: E402
from cerise_sim.rastreador import RastreadorChao, RastreadorImagem  # noqa: E402
from cerise_sim.sintetico import simular_sem_render  # noqa: E402
from cerise_sim.trajetoria import construir  # noqa: E402


# ------------------------------------------------------------------ camera / projecao
def test_projecao_ida_e_volta_nadir_e_obliqua():
    K = Intrinsecos.de_hfov(1280, 960, 70.0)
    rng = np.random.default_rng(1)
    for incl in (0.0, 25.0):
        pose = pose_drone(10.0, -5.0, 40.0, 33.0, incl)
        P = np.column_stack([rng.uniform(-10, 30, 50), rng.uniform(-25, 15, 50), np.zeros(50)])
        uv, z = projetar(P, pose, K)
        assert np.all(z > 0)
        Q = pixel_para_chao(uv, pose, K, 0.0)
        assert np.allclose(Q, P, atol=1e-6)
        # plano elevado (centro da pessoa)
        P2 = P.copy(); P2[:, 2] = 0.85
        uv2, _ = projetar(P2, pose, K)
        assert np.allclose(pixel_para_chao(uv2, pose, K, 0.85), P2, atol=1e-6)


def test_pegada_igual_formula_do_cerise_drones():
    from cerise_drones.frota import Drone
    d = Drone("X", hfov_graus=70.0, vfov_graus=55.0, altitude_m=40.0)
    K = Intrinsecos.de_hfov(1280, 960, 70.0)
    c = pegada_no_chao(pose_drone(0, 0, 40, 90.0), K)
    largura = np.linalg.norm(c[1] - c[0]); comprimento = np.linalg.norm(c[2] - c[1])
    assert largura == pytest.approx(d.largura_pegada_m, rel=1e-9)
    # 4:3 com HFOV 70 da VFOV 55,4 (plano usa 55): diferenca < 1 %
    assert comprimento == pytest.approx(d.comprimento_pegada_m, rel=0.01)


def test_convencao_yaw_topo_da_imagem_na_direcao_de_voo():
    K = Intrinsecos.de_hfov(1000, 750, 70.0)
    pose = pose_drone(0, 0, 40, 0.0)  # voando para LESTE
    (u, v), = projetar([[10.0, 0.0, 0.0]], pose, K)[0]
    assert v < K.cy and u == pytest.approx(K.cx)          # ponto a leste aparece acima do centro
    (u, v), = projetar([[0.0, -10.0, 0.0]], pose, K)[0]
    assert u > K.cx and v == pytest.approx(K.cy)          # ponto ao sul aparece a direita


def test_equivale_a_pixel_para_solo_do_cerise_drones():
    from cerise_drones.agregacao import pixel_para_solo
    K = Intrinsecos.de_hfov(1280, 960, 70.0)
    vfov = K.vfov_graus
    rng = np.random.default_rng(2)
    u = rng.uniform(0, 1280, 30); v = rng.uniform(0, 960, 30)
    for yaw in (0.0, 37.0, 140.0, -100.0):
        Q = pixel_para_chao(np.column_stack([u, v]), pose_drone(5.0, 7.0, 40.0, yaw), K)
        x, y = pixel_para_solo(u, v, 1280, 960, (5.0, 7.0), 40.0, 70.0, vfov, 90.0 - yaw)
        assert np.allclose(Q[:, 0], x, atol=1e-6) and np.allclose(Q[:, 1], y, atol=1e-6)


def test_quaternion_e_convencao_usd():
    R = pose_drone(0, 0, 40, 63.0, 10.0).R_wc
    assert np.allclose(matriz_de_quat(quat_de_matriz(R)), R, atol=1e-9)
    Ru = r_usd_de_cv(pose_drone(0, 0, 40, 0.0).R_wc)
    assert np.allclose(Ru[:, 2], [0, 0, 1])     # camera USD olha para -Z local => +Z local aponta para cima
    assert np.allclose(Ru[:, 1], [1, 0, 0])     # +Y local (topo da imagem) = direcao de voo (leste)
    Rv = olhar_para([0, -100, 50], [0, 0, 0])
    assert np.allclose(Rv[:, 2], np.array([0, 100, -50]) / np.linalg.norm([0, 100, -50]))


def test_intrinsecos_usd():
    K = Intrinsecos.de_hfov(1280, 720, 82.0)
    u = K.usd()
    assert math.degrees(2 * math.atan(u["horizontalAperture"] / (2 * u["focalLength"]))) == pytest.approx(82.0)
    assert u["verticalAperture"] / u["horizontalAperture"] == pytest.approx(720 / 1280)


# ------------------------------------------------------------------ trajetoria
def test_trajetoria_tempo_igual_modelo_cerise_drones():
    from cerise_drones.cobertura import Rota, tempo_basico
    from cerise_drones.frota import Drone
    pts = np.array([[0, 0], [0, 50], [100, 50], [100, 90], [0, 90], [0, 0]], float)
    cobre = np.array([False, True, False, True, False])
    d = Drone("X", velocidade_ms=8.0, altitude_m=30.0)
    tr = construir("X", pts, cobre, 30.0, 8.0, d.vel_subida_ms, d.vel_descida_ms, d.tempo_curva_s)
    assert tr.duracao_s == pytest.approx(tempo_basico(Rota("X", pts, cobre), d), rel=1e-9)
    x, y, z, yaw, cb, fase = tr.amostrar([0.0, 10.0 + 25 / 8.0])
    assert z[0] == 0 and z[1] == pytest.approx(30.0)
    assert np.all(np.abs(np.diff(tr.yaw)) <= 180.0)


# ------------------------------------------------------------------ pessoas
def test_pessoas_dentro_da_area_e_tipos():
    area = Polygon([(0, 0), (300, 0), (300, 200), (150, 260), (0, 200)])
    front = box(0, 0, 150, 260).boundary.intersection(area.buffer(-1))
    ps = gerar_pessoas(area, 60, 120.0, seed=3, fronteiras=front)
    assert len(ps) == 60 and len({p.id for p in ps}) == 60
    tipos = {p.tipo for p in ps}
    assert {"parado", "grupo_parado", "grupo_andando", "caminhante", "cruza_fronteira"} <= tipos
    tab = gabarito(ps, 120.0, 5.0)
    from shapely import contains_xy
    xy = np.array([(r[2], r[3]) for r in tab])
    assert contains_xy(area, xy[:, 0], xy[:, 1]).all()
    for p in ps:
        t, x, y, _ = p.arrays()
        v = np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(t), 1e-9)
        assert v.max() <= 1.75


# ------------------------------------------------------------------ rastreio
def test_rastreador_chao_dois_alvos_cruzando():
    r = RastreadorChao(dist_max_m=1.5, min_hits=3)
    ids_a, ids_b = [], []
    for k in range(40):
        t = k * 0.2
        a = (t * 1.2, 0.0); b = (10 - t * 1.2, 3.0)   # passam a 3 m um do outro
        ids = r.atualizar(t, [a, b], frame=k)
        ids_a.append(ids[0]); ids_b.append(ids[1])
    assert len(set(ids_a)) == 1 and len(set(ids_b)) == 1
    assert len(r.finalizar()) == 2


def test_rastreador_imagem():
    r = RastreadorImagem(min_hits=2)
    for k in range(10):
        r.atualizar([[100 + 3 * k, 50, 110 + 3 * k, 70], [300, 300 - 2 * k, 312, 320 - 2 * k]], frame=k)
    assert len(r.finalizar()) == 2


# ------------------------------------------------------------------ avaliacao
def test_casamento_e_pr():
    gt = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float)
    det = np.array([[1, 1, 11, 11], [200, 200, 210, 210]], float)
    assert casar(det, gt, 0.3) == [(0, 0)]
    pr = pr_quadros({0: (det, np.ones(2), None)}, {0: (["a", "b"], gt, np.zeros(2))}, 0.3)
    assert pr["tp"] == 1 and pr["fp"] == 1 and pr["fn"] == 1


# ------------------------------------------------------------------ ponta a ponta (sem render)
def _cenario_pequeno(tmp_path, n=30, seed=0):
    """Area 240 x 120 m dividida em 2 subareas (x < 120: D1; x >= 120: D2), varredura a 30 m."""
    area = box(0, 0, 240, 120)
    regioes = [box(0, 0, 120, 120), box(120, 0, 240, 120)]
    drones = []
    for i, (x0, x1) in enumerate([(0, 120), (120, 240)]):
        K = Intrinsecos.de_hfov(1280, 960, 70.0)
        larg = 2 * 30 * math.tan(math.radians(35))
        esp = 0.8 * larg
        xs = np.arange(x0 + esp / 2, x1, esp)
        pts = [[x0 + 5, -15.0]]
        for j, x in enumerate(xs):
            pts += [[x, 0.0], [x, 120.0]] if j % 2 == 0 else [[x, 120.0], [x, 0.0]]
        pts.append([x0 + 5, -15.0])
        cobre = [False] + [k % 2 == 0 for k in range(len(pts) - 3)] + [False]
        tr = construir(f"D{i + 1}", np.array(pts), np.array(cobre), 30.0, 6.0)
        drones.append({"id": f"D{i + 1}", "camera": {"intrinsecos": K.para_dict(), "inclinacao_graus": 0.0},
                       "trajetoria": tr.para_dict()})
    dur = max(d["trajetoria"]["t"][-1] for d in drones)
    cen = {"nome": "teste", "origem_enu": {"lon0": 0, "lat0": 0}, "drones": drones, "duracao_s": dur}
    rng = np.random.default_rng(seed)
    pessoas = []
    for k in range(n):
        x, y = rng.uniform(8, 232), rng.uniform(8, 112)
        if k < n // 2:   # paradas
            kf = [(0.0, x, y, 0.0), (dur, x, y, 0.0)]
        else:            # andando devagar
            ang = rng.uniform(0, 2 * np.pi)
            x2 = float(np.clip(x + 0.3 * dur * np.cos(ang), 5, 235)); y2 = float(np.clip(y + 0.3 * dur * np.sin(ang), 5, 115))
            kf = [(0.0, x, y, 0.0), (dur, x2, y2, 0.0)]
        pessoas.append(Pessoa(f"P{k:03d}", "teste", -1, 0.3, kf, []))
    simular_sem_render(cen, pessoas, 5.0, tmp_path)
    return area, regioes, pessoas


def _contar(tmp_path, area, regioes, **kw_oraculo):
    ex = Execucao(tmp_path)
    gts = {c: ex.caixas_gt(c) for c in ex.cameras()}

    def det_por_cam(cam):
        K = ex.intrinsecos(cam)
        o = DetectorOraculo(gts[cam], K.largura, K.altura, **kw_oraculo)
        return lambda q: o(q.frame)

    res = contar_execucao(ex, det_por_cam, area, regioes, ["D1", "D2"], ParamsContagem())
    return res, avaliar_contagem(res, gts, len(ex.pessoas()))


def test_ponta_a_ponta_detector_ideal(tmp_path):
    area, regioes, pessoas = _cenario_pequeno(tmp_path, n=30)
    res, av = _contar(tmp_path, area, regioes)
    ag = res["agregacao"]
    assert ag["soma_deteccoes"] > 5 * len(pessoas)              # soma ingenua super-conta
    assert av["pessoas_vistas_gabarito"] >= 28
    assert abs(av["erro_vs_vistas"]) <= 2                         # contagem unica ~ exata
    assert av["clusters_falsos"] == 0


def test_ponta_a_ponta_detector_ruidoso(tmp_path):
    area, regioes, pessoas = _cenario_pequeno(tmp_path, n=30, seed=1)
    res, av = _contar(tmp_path, area, regioes, p_det=0.8, sigma_px=1.0, fp_por_quadro=0.3, seed=5)
    assert abs(av["erro_pct_vs_vistas"]) <= 20.0
    assert res["agregacao"]["faixas_sem_posse"] >= av["contagem_proposta"]


# ------------------------------------------------------------------ video real (sem pose), formato VisDrone-MOT
def test_contagem_video_visdrone_mot_sintetico(tmp_path):
    sys.path.insert(0, str(SIM / "scripts"))
    from contar_video import contar, ler_visdrone_mot
    rng = np.random.default_rng(0)
    linhas, det = [], {}
    pessoas = [(rng.uniform(50, 1200), rng.uniform(50, 650), rng.uniform(-3, 3), rng.uniform(-2, 2)) for _ in range(12)]
    for f in range(1, 61):
        bx = []
        for i, (x, y, vx, vy) in enumerate(pessoas):
            if i == 0 and f > 30:          # pessoa 0 sai da cena na metade
                continue
            cx, cy = x + vx * f, y + vy * f
            linhas.append(f"{f},{i + 1},{cx - 6:.1f},{cy - 9:.1f},12,18,1,{1 if i % 2 else 2},0,0")
            if rng.random() < 0.9:          # detector com 90 % de revocacao
                bx.append([cx - 6, cy - 9, cx + 6, cy + 9])
        linhas.append(f"{f},99,10,10,30,30,1,4,0,0")  # carro (ignorado)
        det[f] = (np.array(bx).reshape(-1, 4) + rng.normal(0, 0.5, (len(bx), 4)), np.full(len(bx), 0.8))
    arq = tmp_path / "gt.txt"
    arq.write_text(chr(10).join(linhas))
    gt = ler_visdrone_mot(arq)
    assert len(gt[1][0]) == 12 and len(gt[40][0]) == 11
    r = contar(det, gt)
    assert r["pessoas_distintas_gabarito"] == 12
    assert r["faixas_confirmadas"] == 12 and r["faixas_falsas"] == 0 and r["faixas_duplicadas"] == 0
    assert r["pr_iou0.5"]["precisao"] > 0.95 and 0.8 < r["pr_iou0.5"]["revocacao"] < 0.97


# ------------------------------------------------------------------ BoT-SORT (recomendado) + projecao no solo
def test_ap_quadros_perfeito_e_vazio():
    from cerise_sim.avaliacao import ap_quadros
    gt = {0: (["a", "b"], np.array([[10, 10, 20, 30], [50, 50, 60, 70]], float), np.zeros(2))}
    perfeito = {0: (gt[0][1].copy(), np.array([0.9, 0.8]))}
    assert ap_quadros(perfeito, gt, [0], 0.5)["ap"] == pytest.approx(1.0)
    com_fp = {0: (np.vstack([[200, 200, 210, 220], gt[0][1]]), np.array([0.95, 0.9, 0.8]))}
    assert ap_quadros(com_fp, gt, [0], 0.5)["ap"] < 1.0
    assert ap_quadros({}, gt, [0], 0.5)["ap"] == 0.0


def test_gmc_por_pose_mapeia_ponto_do_chao(tmp_path):
    from cerise_sim.botsort_chao import gmc_por_pose
    _cenario_pequeno(tmp_path, n=4)
    ex = Execucao(tmp_path)
    qs = ex.quadros("D1")
    K = ex.intrinsecos("D1")
    H = gmc_por_pose(ex, "D1", 0.0)
    # um quadro em voo de cobertura: um ponto do chao visto no anterior deve cair onde a afim indica
    k = next(i for i in range(1, len(qs)) if qs[i].cobre and qs[i - 1].cobre)
    P = pixel_para_chao(np.array([[640.0, 480.0]]), qs[k - 1].pose, K, 0.0)
    uv, _ = projetar(P, qs[k].pose, K)
    pred = H[qs[k].frame] @ np.array([640.0, 480.0, 1.0])
    assert np.linalg.norm(pred - uv[0]) < 1.0


def test_botsort_no_solo_com_gmc_por_pose(tmp_path):
    pytest.importorskip("ultralytics")
    from cerise_sim.botsort_chao import contar_execucao_botsort, gmc_por_pose
    area, regioes, pessoas = _cenario_pequeno(tmp_path, n=30)
    ex = Execucao(tmp_path)
    gts = {c: ex.caixas_gt(c) for c in ex.cameras()}
    dets, gmcs = {}, {}
    for cam in ["D1", "D2"]:
        K = ex.intrinsecos(cam)
        o = DetectorOraculo(gts[cam], K.largura, K.altura)
        dets[cam] = {q.frame: o(q.frame) for q in ex.quadros(cam)}
        gmcs[cam] = gmc_por_pose(ex, cam, 0.85)
    res = contar_execucao_botsort(ex, dets, gmcs, area, regioes, ["D1", "D2"], ParamsContagem())
    av = avaliar_contagem(res, gts, len(ex.pessoas()))
    assert abs(av["erro_pct_vs_vistas"]) <= 15.0
    assert av["clusters_falsos"] == 0
