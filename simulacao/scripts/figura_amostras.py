"""Figuras de amostras renderizadas: (a) cena vista de cima (visao geral + quadro em nadir de um drone) e
(b) recortes ampliados das cameras dos drones com caixas de gabarito (verde) x trilhas do rastreador
(vermelho), incluindo exemplos de acerto, pessoa perdida e falso positivo.

  python scripts/figura_amostras.py --execucao saidas/run01 --deteccoes resultados/contagem_run01_deteccoes.csv \
      --saida evidencias --prefixo sim_
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cerise_sim.avaliacao import casar, filtrar_gt  # noqa: E402
from cerise_sim.camera import projetar  # noqa: E402
from cerise_sim.dados import Execucao  # noqa: E402

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execucao", required=True)
    ap.add_argument("--deteccoes", required=True)
    ap.add_argument("--saida", default="evidencias")
    ap.add_argument("--prefixo", default="sim_")
    ap.add_argument("--t_cena", type=float, default=None, help="instante da figura da cena (padrao: meio da missao)")
    a = ap.parse_args()
    ex = Execucao(a.execucao)
    out = Path(a.saida); out.mkdir(parents=True, exist_ok=True)
    drones = ex.cameras("drone")
    Q = {c: {q.frame: q for q in ex.quadros(c)} for c in ex.cameras(None)}
    gts = {c: ex.caixas_gt(c) for c in drones}
    det = defaultdict(list)
    for r in csv.DictReader(open(a.deteccoes, encoding="utf-8")):
        det[(r["camera"], int(r["frame"]))].append([float(r[k]) for k in ("x1", "y1", "x2", "y2")])

    # ------------------------------------------------ (a) cena: visao geral + nadir com mais pessoas
    melhor = None
    for c in drones:
        for f, (ids, bx, oc) in gts[c].items():
            ids2, _ = filtrar_gt(ids, bx, oc)
            q = Q[c].get(f)
            if q is not None and q.cobre and q.imagem is not None and (melhor is None or len(ids2) > melhor[0]):
                melhor = (len(ids2), c, f)
    n_p, cam_n, f_n = melhor
    fv = f_n if a.t_cena is None else min(Q["visao_geral"], key=lambda k: abs(Q["visao_geral"][k].t - a.t_cena))
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [16 / 9 * 1.0, 4 / 3 * 1.0]})
    if "visao_geral" in Q and fv in Q["visao_geral"]:
        qv = Q["visao_geral"][fv]
        im = rgb(qv.imagem)
        axs[0].imshow(im)
        Kv = ex.intrinsecos("visao_geral")
        for i, c in enumerate(drones):
            q = Q[c].get(fv)
            if q is None:
                continue
            uv, z = projetar(q.pose.C[None], qv.pose, Kv)
            if z[0] > 0:
                axs[0].plot(uv[0, 0], uv[0, 1], "o", mfc="none", mec=CAT[i % 4], ms=12, mew=2)
                axs[0].annotate(c, (uv[0, 0], uv[0, 1]), xytext=(9, -9), textcoords="offset points", color="white", fontsize=9,
                                fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc=CAT[i % 4], ec="none"))
        axs[0].set_title(f"Visão geral (câmera fixa a sudoeste), t = {qv.t:.1f} s", fontsize=10, loc="left")
    q = Q[cam_n][f_n]
    im = rgb(q.imagem)
    axs[1].imshow(im)
    ids, bx, oc = gts[cam_n][f_n]
    _, bx2 = filtrar_gt(ids, bx, oc)
    for b in bx2:
        axs[1].add_patch(plt.Rectangle((b[0] - 2, b[1] - 2), b[2] - b[0] + 4, b[3] - b[1] + 4, fill=False, ec="#0ca30c", lw=1.0))
    axs[1].set_title(f"Câmera do {cam_n} em nádir ({q.pose.C[2]:.0f} m), t = {q.t:.1f} s: {n_p} pessoas no gabarito (verde)",
                     fontsize=10, loc="left")
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout(); fig.savefig(out / f"{a.prefixo}09_cena_render.png", dpi=150); plt.close(fig)

    # ------------------------------------------------ (b) recortes: acertos, perdidas, falsos positivos
    acertos, perdidas, falsos = [], [], []
    for c in drones:
        for f, q in Q[c].items():
            if not q.cobre or q.imagem is None:
                continue
            ids, bx, oc = gts[c].get(f, ([], np.zeros((0, 4)), np.zeros(0)))
            ids2, g = filtrar_gt(ids, bx, oc)
            d = np.array(det.get((c, f), []), float).reshape(-1, 4)
            pares = casar(d, g, 0.3, 6.0)
            mg = {j for _, j in pares}; md = {i for i, _ in pares}
            for j in range(len(g)):
                (acertos if j in mg else perdidas).append((c, f, g[j], len(g)))
            for i in range(len(d)):
                if i not in md:
                    falsos.append((c, f, d[i], len(g)))
    rng = np.random.default_rng(3)

    def escolhe(lst, n, preferir_muitos=True):
        if not lst:
            return []
        lst = sorted(lst, key=lambda x: -x[3]) if preferir_muitos else lst
        topo = lst[: max(n * 8, n)]
        idx = rng.choice(len(topo), size=min(n, len(topo)), replace=False)
        return [topo[i] for i in idx]
    sel = [("acerto", x) for x in escolhe(acertos, 4)] + [("pessoa perdida", x) for x in escolhe(perdidas, 2)] + \
          [("falso positivo", x) for x in escolhe(falsos, 2, False)]
    if not sel:
        return
    nc = 4
    nl = int(np.ceil(len(sel) / nc))
    fig, axs = plt.subplots(nl, nc, figsize=(3.6 * nc, 3.0 * nl))
    axs = np.atleast_2d(axs)
    R = 70  # meia-janela (px) do recorte
    for k, (tipo, (c, f, b, _n)) in enumerate(sel):
        ax = axs[k // nc, k % nc]
        q = Q[c][f]
        im = rgb(q.imagem)
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        x0, y0 = int(max(0, min(im.shape[1] - 2 * R, cx - R))), int(max(0, min(im.shape[0] - 2 * R, cy - R)))
        ax.imshow(im[y0:y0 + 2 * R, x0:x0 + 2 * R], interpolation="nearest", extent=(x0, x0 + 2 * R, y0 + 2 * R, y0))
        ids, bx, oc = gts[c].get(f, ([], np.zeros((0, 4)), np.zeros(0)))
        _, g = filtrar_gt(ids, bx, oc)
        for gb in g:
            ax.add_patch(plt.Rectangle((gb[0], gb[1]), gb[2] - gb[0], gb[3] - gb[1], fill=False, ec="#0ca30c", lw=1.4))
        for db in det.get((c, f), []):
            ax.add_patch(plt.Rectangle((db[0] - 2, db[1] - 2), db[2] - db[0] + 4, db[3] - db[1] + 4, fill=False, ec="#d03b3b", lw=1.2, ls="--"))
        ax.set_xlim(x0, x0 + 2 * R); ax.set_ylim(y0 + 2 * R, y0)
        ax.set_title(f"{tipo}: {c}, t = {q.t:.1f} s ({b[2] - b[0]:.0f}×{b[3] - b[1]:.0f} px)", fontsize=8, loc="left")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for k in range(len(sel), nl * nc):
        axs[k // nc, k % nc].axis("off")
    fig.suptitle("Recortes de 140×140 px das câmeras dos drones: verde = caixa de gabarito (Replicator); vermelho tracejado = trilha do BoT-SORT calibrado",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(out / f"{a.prefixo}10_deteccoes_vs_gabarito.png", dpi=150); plt.close(fig)
    print("ok", len(acertos), len(perdidas), len(falsos))


if __name__ == "__main__":
    main()
