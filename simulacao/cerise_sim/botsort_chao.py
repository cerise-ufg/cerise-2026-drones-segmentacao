"""Rastreamento recomendado pela validacao em videos reais (validacao_videos/, VisDrone2019-MOT):
YOLO11s ajustado + BoT-SORT do ultralytics com GMC (sparseOptFlow), sem ReID, track_high = new = 0,40,
track_low = 0,05, track_buffer = 30 quadros -- com a CONTAGEM UNICA feita pela posicao no solo, nao pelos IDs.

Fluxo por camera de drone:
  quadros (todos, em ordem) -> deteccoes em cache (conf >= 0,01, NMS 0,6, como no val) + matriz GMC entre
  quadros consecutivos (calculada uma vez sobre as imagens) -> BOTSORT.update() -> cada trilha devolvida
  num quadro de cobertura e projetada no chao com a pose da camera -> Faixa (mesmo tipo do rastreador no
  solo) -> posse da subarea + deduplicacao espaco-temporal (contagem.agregar) -> contagem unica global.

O rastreador recebe todos os quadros (inclusive transito/curvas), para que a GMC seja sempre entre quadros
consecutivos; so as saidas de quadros de cobertura entram nas faixas (ParamsContagem.somente_cobertura).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import shapely

from .contagem import ParamsContagem, agregar, caixas_para_chao
from .rastreador import Faixa


@dataclass
class ParamsBotSort:
    track_high_thresh: float = 0.40
    track_low_thresh: float = 0.05
    new_track_thresh: float = 0.40
    track_buffer: int = 30          # quadros (ultralytics 8.4: max_frames_lost = track_buffer)
    match_thresh: float = 0.8
    fuse_score: bool = True
    gmc_method: str = "sparseOptFlow"
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.8
    with_reid: bool = False


def detectar_e_gmc(execucao, cams, detector, gmc: bool = True, log=print):
    """Roda o detector em TODOS os quadros de cada camera (uma vez) e calcula a GMC entre quadros
    consecutivos. Retorna (dets {cam: {frame: (caixas, scores)}}, gmcs {cam: {frame: H 2x3}}, tempos)."""
    import cv2
    dets, gmcs = {}, {}
    tempos = {"det_s": 0.0, "gmc_s": 0.0, "leitura_s": 0.0, "n": 0}
    for cam in cams:
        dets[cam], gmcs[cam] = {}, {}
        g = None
        if gmc:
            from ultralytics.trackers.utils.gmc import GMC
            g = GMC(method="sparseOptFlow")
        qs = execucao.quadros(cam)
        t_cam = time.perf_counter()
        for i, q in enumerate(qs):
            if q.imagem is None:
                dets[cam][q.frame] = (np.zeros((0, 4)), np.zeros(0))
                gmcs[cam][q.frame] = np.eye(2, 3)
                continue
            t0 = time.perf_counter()
            img = cv2.imread(str(q.imagem))
            t1 = time.perf_counter()
            caixas, scores = detector(img)
            t2 = time.perf_counter()
            H = g.apply(img) if g is not None else np.eye(2, 3)
            t3 = time.perf_counter()
            tempos["leitura_s"] += t1 - t0; tempos["det_s"] += t2 - t1; tempos["gmc_s"] += t3 - t2; tempos["n"] += 1
            dets[cam][q.frame] = (np.asarray(caixas, float).reshape(-1, 4), np.asarray(scores, float).reshape(-1))
            gmcs[cam][q.frame] = np.asarray(H, float)
            if (i + 1) % 200 == 0:
                log(f"  {cam}: {i + 1}/{len(qs)} quadros ({(time.perf_counter() - t_cam) / (i + 1) * 1000:.1f} ms/quadro)")
    return dets, gmcs, tempos


def gmc_por_pose(execucao, cam, z_plano: float = 0.85, grade: int = 9):
    """GMC a partir da telemetria (pose da camera), sem imagem: para cada par de quadros consecutivos,
    projeta uma grade de pixels do quadro anterior no plano z_plano e de volta no quadro atual, e ajusta a
    afim 2x3 (anterior -> atual) por minimos quadrados -- a mesma convencao do GMC do ultralytics."""
    from .camera import pixel_para_chao, projetar
    K = execucao.intrinsecos(cam)
    u, v = np.meshgrid(np.linspace(0.05, 0.95, grade) * K.largura, np.linspace(0.05, 0.95, grade) * K.altura)
    uv = np.column_stack([u.ravel(), v.ravel()])
    out, ant = {}, None
    for q in execucao.quadros(cam):
        H = np.eye(2, 3)
        if ant is not None:
            P = pixel_para_chao(uv, ant.pose, K, z_plano)
            ok = np.isfinite(P).all(axis=1)
            if ok.sum() >= 3:
                uv2, zc = projetar(P[ok], q.pose, K)
                bom = np.isfinite(uv2).all(axis=1) & (zc > 0)
                if bom.sum() >= 3:
                    A = np.column_stack([uv[ok][bom], np.ones(bom.sum())])
                    M, *_ = np.linalg.lstsq(A, uv2[bom], rcond=None)
                    H = M.T
        out[q.frame] = H
        ant = q
    return out


def salvar_cache(caminho, dets, gmcs):
    arrs = {}
    for cam in dets:
        fr = sorted(dets[cam])
        n = [len(dets[cam][f][1]) for f in fr]
        arrs[f"{cam}__frames"] = np.array(fr, int)
        arrs[f"{cam}__n"] = np.array(n, int)
        arrs[f"{cam}__caixas"] = np.concatenate([dets[cam][f][0] for f in fr]) if fr else np.zeros((0, 4))
        arrs[f"{cam}__scores"] = np.concatenate([dets[cam][f][1] for f in fr]) if fr else np.zeros(0)
        arrs[f"{cam}__gmc"] = np.array([gmcs[cam].get(f, np.eye(2, 3)) for f in fr], float).reshape(-1, 2, 3)
    np.savez_compressed(caminho, **arrs)


def carregar_cache(caminho):
    z = np.load(caminho)
    cams = sorted({k.split("__")[0] for k in z.files})
    dets, gmcs = {}, {}
    for cam in cams:
        fr, n = z[f"{cam}__frames"], z[f"{cam}__n"]
        lim = np.concatenate([[0], np.cumsum(n)])
        cx, sc, gm = z[f"{cam}__caixas"], z[f"{cam}__scores"], z[f"{cam}__gmc"]
        dets[cam] = {int(f): (cx[lim[i]:lim[i + 1]], sc[lim[i]:lim[i + 1]]) for i, f in enumerate(fr)}
        gmcs[cam] = {int(f): gm[i] for i, f in enumerate(fr)}
    return dets, gmcs


def novo_botsort(pb: ParamsBotSort):
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml
    cfg = {**YAML.load(check_yaml("botsort.yaml")), **asdict(pb)}
    return BOTSORT(IterableSimpleNamespace(**cfg))


def rastrear_camera_botsort(quadros, K, dets, gmcs, area, p: ParamsContagem, pb: ParamsBotSort):
    """Retorna (faixas confirmadas no solo, log por quadro {frame: (caixas, scores, xy)} das trilhas
    devolvidas pelo BoT-SORT em quadros de cobertura, n de deteccoes de trilha, ids por quadro)."""
    from ultralytics.engine.results import Boxes
    trk = novo_botsort(pb)
    est = {"H": np.eye(2, 3)}
    trk.gmc.apply = lambda img, dets=None: est["H"]  # GMC pre-calculada sobre as imagens (detectar_e_gmc)
    dummy = np.zeros((1, 1, 3), np.uint8)
    area_m = area.buffer(p.margem_area_m) if area is not None else None
    if area_m is not None:
        shapely.prepare(area_m)
    faixas: dict[int, Faixa] = {}
    log, ids_q = {}, {}
    n = 0
    for q in quadros:
        caixas, scores = dets.get(q.frame, (np.zeros((0, 4)), np.zeros(0)))
        est["H"] = gmcs.get(q.frame, np.eye(2, 3))
        data = np.c_[np.asarray(caixas, np.float32).reshape(-1, 4), np.asarray(scores, np.float32).reshape(-1),
                     np.zeros(len(scores), np.float32)]
        r = trk.update(Boxes(data, (K.altura, K.largura)), dummy)
        if p.somente_cobertura and not q.cobre:
            continue
        if len(r):
            tid, bx, sc = r[:, 4].astype(int), r[:, :4].astype(float), r[:, 5].astype(float)
        else:
            tid, bx, sc = np.zeros(0, int), np.zeros((0, 4)), np.zeros(0)
        xy = caixas_para_chao(bx, q.pose, K, p) if len(bx) else np.zeros((0, 2))
        ok = np.isfinite(xy).all(axis=1) if len(xy) else np.zeros(0, bool)
        if area_m is not None and len(xy):
            ok &= shapely.contains_xy(area_m, np.nan_to_num(xy[:, 0]), np.nan_to_num(xy[:, 1]))
        log[q.frame] = (bx[ok], sc[ok], xy[ok])
        ids_q[q.frame] = tid[ok]
        for i in np.nonzero(ok)[0]:
            k = int(tid[i])
            f = faixas.get(k)
            if f is None:
                f = Faixa(k, [], [], [], [], [], xy[i].copy(), np.zeros(2), q.t, 0)
                faixas[k] = f
            f.t.append(q.t); f.xy.append(xy[i].tolist()); f.frames.append(q.frame)
            f.scores.append(float(sc[i])); f.caixas.append(bx[i].tolist())
            f.pos = xy[i].copy(); f.t_ult = q.t; f.hits += 1
            n += 1
    conf = [f for f in faixas.values() if f.hits >= p.min_hits]
    return conf, log, n, ids_q


def contar_execucao_botsort(execucao, dets, gmcs, area, regioes, ids_drones, p: ParamsContagem | None = None,
                            pb: ParamsBotSort | None = None) -> dict:
    p = p or ParamsContagem()
    pb = pb or ParamsBotSort()
    faixas, logs, ndet, tempos, ids_q, brutos = {}, {}, {}, {}, {}, {}
    for cam in ids_drones:
        t0 = time.perf_counter()
        quadros = execucao.quadros(cam)
        K = execucao.intrinsecos(cam)
        faixas[cam], logs[cam], ndet[cam], ids_q[cam] = rastrear_camera_botsort(quadros, K, dets[cam], gmcs[cam], area, p, pb)
        brutos[cam] = len({int(i) for v in ids_q[cam].values() for i in v})
        tempos[cam] = time.perf_counter() - t0
    ag = agregar(faixas, ids_drones, regioes, p)
    ag["soma_deteccoes"] = int(sum(ndet.values()))
    ag["ids_brutos_botsort"] = int(sum(brutos.values()))
    return {"parametros": asdict(p), "parametros_botsort": asdict(pb), "agregacao": ag, "faixas": faixas,
            "logs": logs, "tempo_s": tempos, "ids_por_quadro": ids_q}
