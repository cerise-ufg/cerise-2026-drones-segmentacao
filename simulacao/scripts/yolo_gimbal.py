"""Deteccao de pessoas (YOLO11s ajustado no VisDrone, FP32) nos quadros da camera do drone (gimbal) do close-up,
comparada as caixas de gabarito do Replicator. Desenha verde = gabarito, vermelho = YOLO (com a confianca) e
grava metricas medidas (precisao/revocacao por IoU >= 0,5 no limiar escolhido; AP50 simples).

  python scripts/yolo_gimbal.py --execucao saidas/closeup01 --pesos /w/resultados/best_yolo11s_visdrone_pessoa_960.pt \
      --saida resultados/closeup_yolo_gimbal.json
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def casar(gt, det, lim=0.5):
    """casamento guloso por confianca; devolve lista de (conf, acerto) e n de gabaritos."""
    usados, res = set(), []
    for d in sorted(det, key=lambda d: -d[4]):
        melhor, jm = 0.0, -1
        for j, g in enumerate(gt):
            if j in usados:
                continue
            v = iou(d, g)
            if v > melhor:
                melhor, jm = v, j
        ok = melhor >= lim
        if ok:
            usados.add(jm)
        res.append((d[4], ok))
    return res


def ap(pares, n_gt):
    if not pares or n_gt == 0:
        return 0.0
    pares = sorted(pares, key=lambda p: -p[0])
    tp = np.cumsum([p[1] for p in pares]); fp = np.cumsum([not p[1] for p in pares])
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--execucao", required=True)
    ap_.add_argument("--pesos", required=True)
    ap_.add_argument("--imgsz", type=int, default=1280)
    ap_.add_argument("--conf_min", type=float, default=0.01)
    ap_.add_argument("--conf_desenho", type=float, default=0.25)
    ap_.add_argument("--min_lado_gt", type=float, default=6.0, help="ignora caixas de gabarito menores (px)")
    ap_.add_argument("--saida", required=True)
    a = ap_.parse_args()
    from ultralytics import YOLO
    ex = Path(a.execucao)
    gt = defaultdict(list)
    with open(ex / "gimbal" / "caixas_gt.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            b = [float(r[k]) for k in ("x_min", "y_min", "x_max", "y_max")]
            if max(b[2] - b[0], b[3] - b[1]) >= a.min_lado_gt:
                gt[int(r["quadro"])].append(b)
    quadros = sorted(ex.glob("gimbal/rgb/*.jpg"))
    out_dir = ex / "gimbal" / "anotado"
    out_dir.mkdir(parents=True, exist_ok=True)
    modelo = YOLO(a.pesos)
    nomes = modelo.names
    pares, pares_thr, n_gt_tot, tempos = [], [], 0, []
    por_quadro = []
    for qp in quadros:
        q = int(qp.stem)
        img = cv2.imread(str(qp))
        t0 = time.time()
        r = modelo.predict(img, imgsz=a.imgsz, conf=a.conf_min, iou=0.6, device=0, verbose=False)[0]
        tempos.append(time.time() - t0)
        det = [[*map(float, b), float(c)] for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                                            r.boxes.cls.cpu().numpy()) if "pe" in str(nomes[int(k)]).lower() or len(nomes) == 1]
        g = gt.get(q, [])
        n_gt_tot += len(g)
        pq = casar(g, det)
        pares += pq
        pares_thr += [p for p in pq if p[0] >= a.conf_desenho]
        tp = sum(ok for c, ok in pq if c >= a.conf_desenho)
        nd = sum(1 for c, _ in pq if c >= a.conf_desenho)
        por_quadro.append({"quadro": q, "gt": len(g), "det": nd, "tp": int(tp)})
        for b in g:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60, 200, 60), 2)
        for d in det:
            if d[4] >= a.conf_desenho:
                x0, y0, x1, y1 = map(int, d[:4])
                cv2.rectangle(img, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), (40, 40, 230), 2)
                cv2.putText(img, f"{d[4]:.2f}", (x0 - 3, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 230), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / qp.name), img, [cv2.IMWRITE_JPEG_QUALITY, 94])
    tp = sum(ok for _, ok in pares_thr)
    # por faixa de distancia drone-grupo (quadros.csv: posicao da camera do gimbal e centroide do grupo)
    faixas = []
    qcsv = ex / "quadros.csv"
    if qcsv.exists():
        dist = {}
        for r in csv.DictReader(open(qcsv, encoding="utf-8")):
            v = np.array([float(r["gcx"]) - float(r["centro_x"]), float(r["gcy"]) - float(r["centro_y"]), float(r["gcz"]) - 0.9])
            dist[int(r["quadro"])] = (float(np.linalg.norm(v)), float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))))
        for a0, a1 in [(0, 10), (10, 14), (14, 18), (18, 30)]:
            sel = [d for d in por_quadro if d["quadro"] in dist and a0 <= dist[d["quadro"]][0] < a1]
            if not sel:
                continue
            g_, d_, t_ = sum(d["gt"] for d in sel), sum(d["det"] for d in sel), sum(d["tp"] for d in sel)
            inc = [dist[d["quadro"]][1] for d in sel]
            faixas.append({"dist_m": [a0, a1], "quadros": len(sel), "inclinacao_graus": [round(min(inc), 1), round(max(inc), 1)],
                           "gabarito": g_, "deteccoes": d_, "corretas": t_, "precisao": round(t_ / max(d_, 1), 3),
                           "revocacao": round(t_ / max(g_, 1), 3)})
    res = {
        "execucao": str(ex), "pesos": a.pesos, "classes": {int(k): v for k, v in nomes.items()}, "imgsz": a.imgsz,
        "quadros": len(quadros), "caixas_gabarito": n_gt_tot, "min_lado_gt_px": a.min_lado_gt,
        "ap50": round(ap(pares, n_gt_tot), 4),
        "limiar": a.conf_desenho, "precisao": round(tp / max(len(pares_thr), 1), 4), "revocacao": round(tp / max(n_gt_tot, 1), 4),
        "ms_por_quadro_mediana": round(1000 * float(np.median(tempos)), 1) if tempos else None,
        "por_faixa_distancia": faixas,
        "por_quadro": por_quadro,
    }
    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "por_quadro"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
