"""Graficos da deteccao/contagem (a partir de resultados/deteccao/*.json|csv trazidos do Spark).

Uso: python scripts/graficos_deteccao.py
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from cerise_drones.visualizar import TINTA, TINTA2, estilo  # noqa: E402

D = RAIZ / "resultados" / "deteccao"; EV = RAIZ / "evidencias"
estilo()

# ---------- curvas de treino
r = pd.read_csv(D / "results.csv")
r.columns = [c.strip() for c in r.columns]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6))
for col, lb, cor in [("train/box_loss", "treino - caixa", "#2a78d6"), ("train/cls_loss", "treino - classe", "#eb6834"),
                     ("val/box_loss", "validacao - caixa", "#1baf7a"), ("val/cls_loss", "validacao - classe", "#6250d6")]:
    axs[0].plot(r.epoch, r[col], marker="o", ms=5, lw=2, color=cor, label=lb)
axs[0].set_xlabel("epoca"); axs[0].set_ylabel("perda"); axs[0].legend(fontsize=7.5)
axs[0].set_title("(a) Perdas por epoca", loc="left", fontsize=9, color=TINTA)
axs[1].plot(r.epoch, r["metrics/mAP50(B)"], marker="o", ms=5, lw=2, color="#2a78d6", label="mAP50 (pessoa)")
axs[1].plot(r.epoch, r["metrics/mAP50-95(B)"], marker="s", ms=5, lw=2, color="#eb6834", label="mAP50-95 (pessoa)")
for x, y in zip(r.epoch, r["metrics/mAP50(B)"]):
    axs[1].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7, color=TINTA2)
axs[1].set_xlabel("epoca"); axs[1].set_ylim(0, 0.65); axs[1].legend(fontsize=7.5)
axs[1].set_title("(b) Validacao (VisDrone2019-DET val, 548 imagens)", loc="left", fontsize=9, color=TINTA)
for ax in axs:
    ax.set_xticks(r.epoch)
fig.suptitle("Fine-tuning YOLO11s, 960 px, 48,7 min no GB10 (epoca 5 encerrada pelo limite de tempo)", x=0.01, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(EV / "13_deteccao_treino.png", bbox_inches="tight")
plt.close(fig)

av = D / "avaliacao_deteccao.json"
coco = D / "val_coco_pretreinado.json"
if av.exists() and coco.exists():
    A = json.load(open(av)); C = json.load(open(coco))
    # ---------- mAP: pre-treinados x ajustado
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.0), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    ax = axs[0]
    lab, m50, m95 = [], [], []
    for c in C:
        lab.append(f"{c['modelo'][:-3]}\nCOCO {c['imgsz']}"); m50.append(c["mAP50"]); m95.append(c["mAP50_95"])
    ft = [m for m in A["metricas"] if m["split"] == "val" and "ajustado" in m["modelo"]][0]
    lab.append("yolo11s\najustado 960"); m50.append(ft["mAP50"]); m95.append(ft["mAP50_95"])
    x = np.arange(len(lab))
    ax.bar(x - 0.2, m50, 0.38, color="#2a78d6", label="mAP50")
    ax.bar(x + 0.2, m95, 0.38, color="#eb6834", label="mAP50-95")
    for i, v in enumerate(m50):
        ax.text(i - 0.2, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5, color=TINTA)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=6.8)
    ax.set_title("(a) Classe pessoa no VisDrone val (pedestrian+people)", loc="left", fontsize=9, color=TINTA)
    ax.legend(fontsize=7.5)
    # ---------- MAE x limiar
    ax = axs[1]
    cores = {"yolo11s COCO (pre-treinado)": "#eb6834", "yolo11s ajustado VisDrone-pessoa": "#2a78d6"}
    for nome, c in A["contagem"].items():
        cv = pd.DataFrame(c["curva_val"]); ct = pd.DataFrame(c["curva_test"])
        ax.plot(cv.limiar, cv.MAE_mascarada, color=cores[nome], lw=2, label=f"{nome.split(' (')[0].replace('VisDrone-pessoa', '')} - val")
        ax.plot(ct.limiar, ct.MAE_mascarada, color=cores[nome], lw=1.5, ls="--", label="  idem - test-dev")
        ax.axvline(c["limiar_escolhido_no_val"], color=cores[nome], lw=0.8, ls=":")
    ax.set_xlabel("limiar de confianca"); ax.set_ylabel("MAE da contagem (pessoas/imagem)")
    ax.set_title("(b) Erro de contagem x limiar", loc="left", fontsize=9, color=TINTA)
    ax.legend(fontsize=6.8)
    # ---------- dispersao test-dev
    ax = axs[2]
    for nome, c in A["contagem"].items():
        dd = np.array(c["dispersao_test"])
        ax.scatter(dd[:, 0], dd[:, 1], s=7, alpha=0.45, color=cores[nome], label=nome.split(" (")[0])
    mx = max(np.array(c["dispersao_test"]).max() for c in A["contagem"].values())
    ax.plot([0, mx], [0, mx], color=TINTA2, lw=1, ls="--")
    ax.set_xlabel("pessoas anotadas na imagem"); ax.set_ylabel("pessoas detectadas")
    ax.set_title("(c) Contagem por imagem - test-dev (1610 imgs)", loc="left", fontsize=9, color=TINTA)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(EV / "14_deteccao_metricas.png", bbox_inches="tight")
    plt.close(fig)

# ---------- mosaico de exemplos
ex = sorted((D / "exemplos").glob("*.jpg"))
if ex:
    ims = [Image.open(f) for f in ex[:4]]
    w = 1280
    hs = [int(im.height * w / im.width) for im in ims]
    H = max(hs)
    mos = Image.new("RGB", (2 * w + 10, 2 * H + 10), "white")
    for k, im in enumerate(ims):
        im = im.resize((w, int(im.height * w / im.width)))
        mos.paste(im, ((k % 2) * (w + 10), (k // 2) * (H + 10)))
    mos = mos.resize((1600, int(mos.height * 1600 / mos.width)))
    mos.save(EV / "15_deteccao_exemplos.png", optimize=True)
print("ok")
