"""Avaliacao contra o gabarito do simulador.

- Por quadro: precisao/revocacao das deteccoes contra as caixas 2D de gabarito (Replicator
  bounding_box_2d_tight), casamento Hungaro por IoU (limiares 0,3 e 0,5) e, para alvos minusculos,
  tambem por distancia entre centros; erro absoluto da contagem por quadro.
- Contagem unica global: cada faixa recebe o rotulo (pessoa_id) majoritario das suas deteccoes
  (casadas com as caixas de gabarito do mesmo quadro); cada cluster da agregacao recebe o rotulo
  majoritario das faixas. Decomposicao: unicos corretos + duplicatas + falsos (sem rotulo);
  perdidas = pessoas vistas no gabarito que nao viraram cluster.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from .rastreador import iou_matriz


def casar(det, gt, iou_min: float = 0.3, dist_centro_px: float | None = None):
    """Retorna lista de pares (i_det, j_gt)."""
    det = np.asarray(det, float).reshape(-1, 4); gt = np.asarray(gt, float).reshape(-1, 4)
    if len(det) == 0 or len(gt) == 0:
        return []
    iou = iou_matriz(det, gt)
    ok = iou >= iou_min
    if dist_centro_px is not None:
        cd = np.column_stack([(det[:, 0] + det[:, 2]) / 2, (det[:, 1] + det[:, 3]) / 2])
        cg = np.column_stack([(gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2])
        dist = np.linalg.norm(cd[:, None] - cg[None], axis=2)
        ok |= dist <= dist_centro_px
    C = np.where(ok, 1 - iou, 1e6)
    li, co = linear_sum_assignment(C)
    return [(int(i), int(j)) for i, j in zip(li, co) if C[i, j] < 1e6]


def filtrar_gt(ids, caixas, oclusao, area_min_px: float = 4.0, oclusao_max: float = 0.9):
    caixas = np.asarray(caixas, float).reshape(-1, 4)
    a = (caixas[:, 2] - caixas[:, 0]) * (caixas[:, 3] - caixas[:, 1])
    m = (a >= area_min_px) & (np.asarray(oclusao, float) <= oclusao_max)
    return [i for i, k in zip(ids, m) if k], caixas[m]


def pr_quadros(log_det: dict, caixas_gt: dict, iou_min: float = 0.3, dist_centro_px: float | None = None,
               area_min_px: float = 4.0, oclusao_max: float = 0.9) -> dict:
    tp = fp = fn = 0
    erros = []
    for f, (caixas, scores, _xy) in log_det.items():
        ids, gt = filtrar_gt(*caixas_gt.get(f, ([], np.zeros((0, 4)), np.zeros(0))), area_min_px, oclusao_max)
        pares = casar(caixas, gt, iou_min, dist_centro_px)
        tp += len(pares)
        fp += len(caixas) - len(pares)
        fn += len(gt) - len(pares)
        erros.append(abs(len(caixas) - len(gt)))
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    return {"quadros": len(log_det), "tp": tp, "fp": fp, "fn": fn, "precisao": P, "revocacao": R,
            "f1": 2 * P * R / max(P + R, 1e-9), "mae_contagem_quadro": float(np.mean(erros)) if erros else None,
            "iou_min": iou_min, "dist_centro_px": dist_centro_px}


def rotular_faixa(faixa, caixas_gt: dict, iou_min: float = 0.1, dist_centro_px: float = 6.0):
    votos = Counter()
    for f, b in zip(faixa.frames, faixa.caixas):
        ids, gt, _oc = caixas_gt.get(f, ([], np.zeros((0, 4)), np.zeros(0)))
        if not len(ids):
            continue
        pares = casar([b], gt, iou_min, dist_centro_px)
        if pares:
            votos[ids[pares[0][1]]] += 1
    if not votos:
        return None, 0.0
    pid, n = votos.most_common(1)[0]
    return pid, n / max(len(faixa.frames), 1)


def pessoas_vistas(caixas_gt_por_cam: dict, quadros_usados: dict, area_min_px: float = 4.0, oclusao_max: float = 0.9) -> set:
    vistos = set()
    for cam, gt in caixas_gt_por_cam.items():
        usados = quadros_usados.get(cam)
        for f, (ids, bx, oc) in gt.items():
            if usados is not None and f not in usados:
                continue
            ids2, _ = filtrar_gt(ids, bx, oc, area_min_px, oclusao_max)
            vistos |= set(ids2)
    return vistos


def avaliar_contagem(resultado: dict, caixas_gt_por_cam: dict, n_pessoas_cena: int) -> dict:
    """resultado = contagem.contar_execucao(...)."""
    faixas = resultado["faixas"]
    rot = {}
    for cam, fs in faixas.items():
        for f in fs:
            rot[(cam, f.id)] = rotular_faixa(f, caixas_gt_por_cam.get(cam, {}))[0]
    usados = {cam: set(log.keys()) for cam, log in resultado["logs"].items()}
    vistos = pessoas_vistas(caixas_gt_por_cam, usados)
    ag = resultado["agregacao"]
    labels = []
    for cl in ag.get("clusters", []):
        v = Counter(rot.get(tuple(m)) for m in cl)
        labels.append(v.most_common(1)[0][0] if v else None)
    corretos = len({l for l in labels if l is not None})
    falsos = sum(1 for l in labels if l is None)
    dup = sum(1 for l in labels if l is not None) - corretos
    est = ag.get("proposto", 0)
    out = {
        "pessoas_na_cena": n_pessoas_cena,
        "pessoas_vistas_gabarito": len(vistos),
        "contagem_proposta": est,
        "erro_vs_cena": est - n_pessoas_cena,
        "erro_pct_vs_cena": 100.0 * (est - n_pessoas_cena) / max(n_pessoas_cena, 1),
        "erro_vs_vistas": est - len(vistos),
        "erro_pct_vs_vistas": 100.0 * (est - len(vistos)) / max(len(vistos), 1),
        "clusters_corretos_unicos": corretos,
        "clusters_duplicados": dup,
        "clusters_falsos": falsos,
        "pessoas_perdidas": len(vistos - {l for l in labels if l is not None}),
        "faixas_rotuladas": sum(1 for v in rot.values() if v is not None),
        "faixas_total": len(rot),
        "outras_estrategias": {
            "soma_deteccoes": ag.get("soma_deteccoes"),
            "faixas_sem_posse": ag.get("faixas_sem_posse"),
            "cerise_drones_agregacao": {k: v for k, v in ag.get("cerise_drones_agregacao", {}).items()
                                        if k in ("soma_ingenua", "dedup_por_drone_sem_posse", "proposto_posse_dedup_borda", "erro")},
        },
    }
    return out
