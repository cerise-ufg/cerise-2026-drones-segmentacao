"""Video para o relatorio: mosaico (visao geral + cameras dos drones) com caixas de gabarito (verde),
deteccoes rastreadas (vermelho, com id do cluster) e painel com a contagem acumulada.
Funciona tambem sem imagens (execucao sintetica sem render: fundo cinza), para testar o pipeline.
Uso (conteiner cerise26_drones_sim):
  python scripts/video_relatorio.py --execucao saidas/run01 --deteccoes resultados/contagem_run01_deteccoes.csv \
      --saida evidencias/video_run01.mp4 [--fps 5] [--max_quadros 600]
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cerise_sim.dados import Execucao  # noqa: E402

CW, CH = 640, 480
CORES = {"D1": (180, 119, 31), "D2": (40, 39, 214), "D3": (44, 160, 44), "D4": (189, 103, 148)}  # BGR


def celula(img, titulo, caixas_gt=(), deteccoes=(), escala=None):
    h, w = img.shape[:2]
    s = min(CW / w, CH / h) if escala is None else escala
    r = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    out = np.full((CH, CW, 3), 30, np.uint8)
    y0, x0 = (CH - r.shape[0]) // 2, (CW - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    for b in caixas_gt:
        p1 = (int(x0 + b[0] * s), int(y0 + b[1] * s)); p2 = (int(x0 + b[2] * s), int(y0 + b[3] * s))
        cv2.rectangle(out, p1, p2, (0, 220, 0), 1)
    for (b, cl) in deteccoes:
        p1 = (int(x0 + b[0] * s) - 2, int(y0 + b[1] * s) - 2); p2 = (int(x0 + b[2] * s) + 2, int(y0 + b[3] * s) + 2)
        cv2.rectangle(out, p1, p2, (0, 0, 255), 1)
        if cl >= 0:
            cv2.putText(out, str(cl), (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.rectangle(out, (0, 0), (CW, 22), (0, 0, 0), -1)
    cv2.putText(out, titulo, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execucao", required=True)
    ap.add_argument("--deteccoes", default=None)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--max_quadros", type=int, default=None)
    a = ap.parse_args()
    ex = Execucao(a.execucao)
    drones = ex.cameras("drone")[:4]
    quadros = {c: {q.frame: q for q in ex.quadros(c)} for c in ex.cameras(None)}
    gts = {c: ex.caixas_gt(c) for c in drones}
    det = defaultdict(list)
    t_cluster = {}
    if a.deteccoes:
        for r in csv.DictReader(open(a.deteccoes, encoding="utf-8")):
            det[(r["camera"], int(r["frame"]))].append(([float(r[k]) for k in ("x1", "y1", "x2", "y2")], int(r["cluster"])))
            cl, t = int(r["cluster"]), float(r["t"])
            if cl >= 0:
                t_cluster[cl] = min(t, t_cluster.get(cl, 1e9))
    frames = sorted(set().union(*[set(v) for v in quadros.values()]))
    if a.max_quadros:
        frames = frames[: a.max_quadros]
    fps = a.fps or ex.fps
    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(a.saida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (3 * CW, 2 * CH))
    vistos = set()
    for f in frames:
        cel = []
        t = None
        for c in ["visao_geral"] + drones:
            q = quadros.get(c, {}).get(f)
            if q is None:
                cel.append(np.full((CH, CW, 3), 30, np.uint8))
                continue
            t = q.t
            K = ex.intrinsecos(c)
            img = cv2.imread(str(q.imagem)) if q.imagem else None
            if img is None:
                img = np.full((K.altura, K.largura, 3), 90, np.uint8)
            g = gts.get(c, {}).get(f, ([], np.zeros((0, 4)), None))
            vistos |= set(g[0])
            tit = "visao geral" if c == "visao_geral" else f"{c}  z={q.pose.C[2]:.0f} m  det={len(det.get((c, f), []))}  gab={len(g[0])}"
            cel.append(celula(img, tit, g[1], det.get((c, f), [])))
        painel = np.full((CH, CW, 3), 245, np.uint8)
        n_cl = sum(1 for v in t_cluster.values() if t is not None and v <= t)
        linhas = [f"t = {t if t is not None else 0:.1f} s (quadro {f})", "",
                  f"pessoas distintas ja vistas (gabarito): {len(vistos)}",
                  f"contagem unica estimada (clusters): {n_cl}" if a.deteccoes else "sem deteccoes (so gabarito)",
                  "", "verde: caixa de gabarito (simulador)", "vermelho: deteccao rastreada (id do cluster)"]
        for i, ln in enumerate(linhas):
            cv2.putText(painel, ln, (16, 50 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1)
        while len(cel) < 5:
            cel.append(np.full((CH, CW, 3), 30, np.uint8))
        grade = np.vstack([np.hstack(cel[0:3]), np.hstack([cel[3], cel[4], painel])])
        vw.write(grade)
    vw.release()
    print(a.saida, len(frames), "quadros")


if __name__ == "__main__":
    main()
