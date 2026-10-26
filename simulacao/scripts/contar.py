"""Pipeline de contagem sobre uma execucao gravada (Isaac Sim, Gazebo ou sintetica sem render).

YOLO11s ajustado (FP32) no conteiner da frente de drones (ver scripts/rodar_contagem_spark.sh):
  python scripts/contar.py --execucao saidas/run01 --cenario cenario/reduzido --detector yolo \
      --pesos /w/resultados/best_yolo11s_visdrone_pessoa_960.pt --imgsz 1280 --saida resultados/contagem_run01.json
Detector oraculo (gabarito + ruido; isola o erro da agregacao):
  python scripts/contar.py --execucao cenario/reduzido/sem_render --cenario cenario/reduzido --detector oraculo \
      --p_det 0.8 --sigma_px 1.0 --fp 0.3 --saida resultados/contagem_oraculo.json
Gera tambem <saida>_deteccoes.csv (deteccoes associadas a faixas e clusters) e, com --figura, um mapa PNG.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

AQUI = Path(__file__).resolve().parent
SIM = AQUI.parent
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM.parent))

import numpy as np  # noqa: E402
from shapely.geometry import MultiPolygon, Polygon  # noqa: E402

from cerise_sim.avaliacao import avaliar_contagem, pr_quadros  # noqa: E402
from cerise_sim.contagem import DetectorOraculo, DetectorYOLO, ParamsContagem, contar_execucao  # noqa: E402
from cerise_sim.dados import Execucao  # noqa: E402


def poli(lista):
    ps = [Polygon(c) for c in lista]
    return ps[0] if len(ps) == 1 else MultiPolygon(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execucao", required=True)
    ap.add_argument("--cenario", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--detector", choices=["yolo", "oraculo"], default="yolo")
    ap.add_argument("--pesos", default="resultados/deteccao/best_yolo11s_visdrone_pessoa_960.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.175)
    ap.add_argument("--device", default="0")
    ap.add_argument("--p_det", type=float, default=1.0)
    ap.add_argument("--sigma_px", type=float, default=0.0)
    ap.add_argument("--fp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figura", action="store_true")
    for f in fields(ParamsContagem):
        ap.add_argument("--" + f.name, type=type(f.default) if not isinstance(f.default, bool) else (lambda s: s.lower() in ("1", "true", "sim")),
                        default=f.default)
    a = ap.parse_args()
    p = ParamsContagem(**{f.name: getattr(a, f.name) for f in fields(ParamsContagem)})
    cen = json.loads((Path(a.cenario) / "cenario.json").read_text(encoding="utf-8"))
    area = poli(cen["area_enu"])
    ex = Execucao(a.execucao)
    ids = [d for d in cen["regioes_enu"] if d in ex.cameras("drone")]
    regioes = [poli(cen["regioes_enu"][d]) for d in ids]
    gts = {c: ex.caixas_gt(c) for c in ids}
    t_det = {"s": 0.0, "n": 0}
    if a.detector == "yolo":
        dev = int(a.device) if a.device.isdigit() else a.device
        yolo = DetectorYOLO(a.pesos, a.imgsz, a.conf, device=dev)

        def fabrica(cam):
            def det(q):
                if q.imagem is None:
                    return np.zeros((0, 4)), np.zeros(0)
                t0 = time.perf_counter()
                r = yolo(q.imagem)
                t_det["s"] += time.perf_counter() - t0; t_det["n"] += 1
                return r
            return det
    else:
        def fabrica(cam):
            K = ex.intrinsecos(cam)
            o = DetectorOraculo(gts[cam], K.largura, K.altura, a.p_det, a.sigma_px, a.fp, seed=a.seed + sum(map(ord, cam)))
            return lambda q: o(q.frame)
    t0 = time.time()
    res = contar_execucao(ex, fabrica, area, regioes, ids, p)
    av = avaliar_contagem(res, gts, len(ex.pessoas()))
    pr = {}
    for nome, kw in {"iou0.5": dict(iou_min=0.5), "iou0.3": dict(iou_min=0.3), "iou0.3_ou_centro6px": dict(iou_min=0.3, dist_centro_px=6.0)}.items():
        tot = {"tp": 0, "fp": 0, "fn": 0}
        por_cam = {}
        for cam in ids:
            r = pr_quadros(res["logs"][cam], gts[cam], **kw)
            por_cam[cam] = r
            for k in tot:
                tot[k] += r[k]
        P = tot["tp"] / max(tot["tp"] + tot["fp"], 1); R = tot["tp"] / max(tot["tp"] + tot["fn"], 1)
        pr[nome] = {**tot, "precisao": P, "revocacao": R, "f1": 2 * P * R / max(P + R, 1e-9), "por_camera": por_cam}
    ag = res["agregacao"]
    saida = {
        "execucao": str(a.execucao), "cenario": cen["nome"], "simulador": ex.meta.get("simulador"),
        "detector": a.detector, "detector_params": ({"pesos": a.pesos, "imgsz": a.imgsz, "conf": a.conf, "fp32": True}
                                                     if a.detector == "yolo" else {"p_det": a.p_det, "sigma_px": a.sigma_px, "fp_por_quadro": a.fp}),
        "parametros_contagem": asdict(p), "drones": ids, "fps": ex.fps,
        "avaliacao_contagem": av, "deteccao_por_quadro": pr,
        "agregacao": {k: v for k, v in ag.items() if k != "clusters"},
        "faixas_por_drone": {c: len(fs) for c, fs in res["faixas"].items()},
        "quadros_usados_por_drone": {c: len(lg) for c, lg in res["logs"].items()},
        "tempo_total_s": time.time() - t0,
        "fps_detector": (t_det["n"] / t_det["s"]) if t_det["s"] > 0 else None,
    }
    out = Path(a.saida)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(saida, indent=1, default=float), encoding="utf-8")
    # deteccoes associadas
    cl_de = {}
    for k, cl in enumerate(ag.get("clusters", [])):
        for m in cl:
            cl_de[tuple(m)] = k
    with open(out.with_name(out.stem + "_deteccoes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["camera", "frame", "t", "faixa", "cluster", "x1", "y1", "x2", "y2", "X", "Y", "score"])
        for cam, fs in res["faixas"].items():
            for fx in fs:
                for fr, t, b, xy, s in zip(fx.frames, fx.t, fx.caixas, fx.xy, fx.scores):
                    w.writerow([cam, fr, round(t, 3), fx.id, cl_de.get((cam, fx.id), -1), *np.round(b, 1), *np.round(xy, 2), round(s, 3)])
    print(json.dumps({k: saida[k] for k in ("avaliacao_contagem", "agregacao", "tempo_total_s", "fps_detector")}, indent=1, default=float))
    if a.figura:
        figura(cen, ex, res, av, out.with_suffix(".png"), a)


def figura(cen, ex, res, av, caminho, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MP
    from cerise_sim.avaliacao import rotular_faixa
    cores = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for i, (d, polys) in enumerate(cen["regioes_enu"].items()):
        for pl in polys:
            ax.add_patch(MP(pl, closed=True, fc=cores[i % 4], alpha=0.12, ec=cores[i % 4], lw=1))
    gab = ex.gabarito()
    ult = gab[max(gab)]
    xy = np.array(list(ult.values()))
    ax.plot(xy[:, 0], xy[:, 1], "k.", ms=4, label=f"pessoas (gabarito, posicao final) N={len(ult)}")
    gts = {c: ex.caixas_gt(c) for c in res["faixas"]}
    rot = {(c, f.id): rotular_faixa(f, gts[c])[0] for c, fs in res["faixas"].items() for f in fs}
    vistos = set()
    marc = {"unico": ("o", "#2ca02c"), "dup": ("s", "#ff7f0e"), "falso": ("x", "#d62728")}
    usados = set()
    for cl in res["agregacao"].get("clusters", []):
        labs = [rot.get(tuple(m)) for m in cl]
        lab = max(set(labs), key=labs.count)
        fx = [f for c, fs in res["faixas"].items() for f in fs if (c, f.id) in {tuple(m) for m in cl}]
        P = np.mean([f.posicao_representativa() for f in fx], axis=0)
        tipo = "falso" if lab is None else ("dup" if lab in vistos else "unico")
        vistos.add(lab)
        mk, c = marc[tipo]
        ax.plot(P[0], P[1], mk, mfc="none" if tipo == "unico" else c, mec=c, ms=7,
                label={"unico": "cluster correto", "dup": "duplicata", "falso": "falso positivo"}[tipo] if tipo not in usados else None)
        usados.add(tipo)
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("x leste (m)"); ax.set_ylabel("y norte (m)")
    ax.set_title(f"Contagem unica global ({a.detector}): estimada {av['contagem_proposta']} x vistas {av['pessoas_vistas_gabarito']} "
                 f"(cena {av['pessoas_na_cena']}); dup {av['clusters_duplicados']}, falsos {av['clusters_falsos']}, perdidas {av['pessoas_perdidas']}",
                 fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(caminho, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
