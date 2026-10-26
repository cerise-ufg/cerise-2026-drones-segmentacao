"""Contagem de pessoas em VIDEO REAL de drone, sem pose da camera (modo imagem): YOLO11s ajustado (FP32)
-> rastreio no plano da imagem (RastreadorImagem) -> contagem de faixas confirmadas. Com anotacao no formato
VisDrone2019-MOT (frame,id,x,y,w,h,score,categoria,truncamento,oclusao; pessoa = categorias 1 e 2), mede
precisao/revocacao por quadro, MAE da contagem por quadro e o erro da contagem unica (ids distintos).

  python scripts/contar_video.py --quadros <pasta com 0000001.jpg...> --gt <anotacao.txt> --pesos best.pt --saida r.json
  python scripts/contar_video.py --video voo.mp4 --pesos best.pt --saida r.json --passo 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))

import numpy as np  # noqa: E402

from cerise_sim.avaliacao import pr_quadros, rotular_faixa  # noqa: E402
from cerise_sim.rastreador import RastreadorImagem  # noqa: E402

CATEGORIAS_PESSOA = {1, 2}


def ler_visdrone_mot(arq) -> dict:
    """frame -> (ids, caixas xyxy, oclusao[0..1]); so pedestre/pessoas (cat 1, 2)."""
    out: dict = {}
    for ln in Path(arq).read_text().splitlines():
        v = ln.strip().split(",")
        if len(v) < 8:
            continue
        f, pid, x, y, w, h = int(v[0]), v[1], float(v[2]), float(v[3]), float(v[4]), float(v[5])
        if int(v[7]) not in CATEGORIAS_PESSOA:
            continue
        ocl = {0: 0.0, 1: 0.25, 2: 0.75}.get(int(v[9]), 0.0) if len(v) > 9 else 0.0
        ids, bx, oc = out.setdefault(f, ([], [], []))
        ids.append(pid); bx.append([x, y, x + w, y + h]); oc.append(ocl)
    return {f: (ids, np.array(bx, float).reshape(-1, 4), np.array(oc, float)) for f, (ids, bx, oc) in out.items()}


def quadros_de(a):
    if a.quadros:
        fs = sorted(p for p in Path(a.quadros).iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
        for i, p in enumerate(fs):
            if i % a.passo == 0:
                yield int(p.stem) if p.stem.isdigit() else i + 1, p
    else:
        import cv2
        cap = cv2.VideoCapture(a.video)
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            i += 1
            if (i - 1) % a.passo == 0:
                yield i, img[..., ::-1].copy()  # BGR -> RGB


def contar(deteccoes_por_quadro: dict, gt: dict | None, min_hits: int = 3, max_idade: int = 10) -> dict:
    """deteccoes_por_quadro: frame -> (caixas, scores). Separado do YOLO para ser testavel."""
    rast = RastreadorImagem(min_hits=min_hits, max_idade_quadros=max_idade)
    for f in sorted(deteccoes_por_quadro):
        b, s = deteccoes_por_quadro[f]
        rast.atualizar(b, s, frame=f)
    faixas = rast.finalizar()
    out = {"faixas_confirmadas": len(faixas), "quadros": len(deteccoes_por_quadro)}
    if gt is not None:
        gt_usado = {f: gt.get(f, ([], np.zeros((0, 4)), np.zeros(0))) for f in deteccoes_por_quadro}
        ids_vistos = {i for f in gt_usado for i in gt_usado[f][0]}
        rot = [rotular_faixa(fx, gt_usado, iou_min=0.3, dist_centro_px=8.0)[0] for fx in faixas]
        corretos = {r for r in rot if r is not None}
        log = {f: (np.asarray(b), np.asarray(s), None) for f, (b, s) in deteccoes_por_quadro.items()}
        out.update({
            "pessoas_distintas_gabarito": len(ids_vistos),
            "erro_contagem_unica": len(faixas) - len(ids_vistos),
            "erro_pct_contagem_unica": 100.0 * (len(faixas) - len(ids_vistos)) / max(len(ids_vistos), 1),
            "faixas_corretas_unicas": len(corretos),
            "faixas_duplicadas": sum(1 for r in rot if r is not None) - len(corretos),
            "faixas_falsas": sum(1 for r in rot if r is None),
            "pessoas_perdidas": len(ids_vistos - corretos),
            "pr_iou0.5": pr_quadros(log, gt_usado, 0.5, area_min_px=0, oclusao_max=1.0),
            "pr_iou0.3": pr_quadros(log, gt_usado, 0.3, area_min_px=0, oclusao_max=1.0),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quadros")
    g.add_argument("--video")
    ap.add_argument("--gt", default=None, help="anotacao VisDrone2019-MOT do video")
    ap.add_argument("--pesos", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.175)
    ap.add_argument("--passo", type=int, default=1)
    ap.add_argument("--min_hits", type=int, default=3)
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    from cerise_sim.contagem import DetectorYOLO
    yolo = DetectorYOLO(a.pesos, a.imgsz, a.conf)
    det = {}
    t0 = time.perf_counter()
    for f, img in quadros_de(a):
        det[f] = yolo(img)
    t_det = time.perf_counter() - t0
    gt = ler_visdrone_mot(a.gt) if a.gt else None
    r = contar(det, gt, a.min_hits, max_idade=max(2, 10 // a.passo))
    r.update({"fonte": a.quadros or a.video, "pesos": a.pesos, "imgsz": a.imgsz, "conf": a.conf, "passo": a.passo,
              "fps_detector": len(det) / max(t_det, 1e-9), "deteccoes_por_quadro_media": float(np.mean([len(b) for b, _ in det.values()])) if det else 0})
    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida).write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in r.items() if not k.startswith("pr_")}, indent=1, default=float))


if __name__ == "__main__":
    main()
