"""Mosaico com o quadro central de cada sequencia do VisDrone2019-MOT (val e test-dev), com nome, resolucao e
atributos medidos (pessoas/quadro, luminancia). Saida: evidencias/v00_mosaico_sequencias.png"""
import glob
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MOT = "/w/mot"
cond = {(c["split"], c["seq"]): c for c in json.load(open("/w/validacao_videos/resultados/condicoes_sequencias.json"))}
itens = [(s, os.path.basename(d)) for s in ("val", "test-dev") for d in sorted(glob.glob(f"{MOT}/VisDrone2019-MOT-{s}/sequences/*"))]
nc = 6; nl = (len(itens) + nc - 1) // nc
fig, axs = plt.subplots(nl, nc, figsize=(nc * 3.2, nl * 2.2))
for ax in axs.ravel():
    ax.axis("off")
for ax, (s, q) in zip(axs.ravel(), itens):
    arqs = sorted(glob.glob(f"{MOT}/VisDrone2019-MOT-{s}/sequences/{q}/*.jpg"))
    img = cv2.imread(arqs[len(arqs) // 2])
    img = cv2.resize(img, (480, int(480 * img.shape[0] / img.shape[1])))
    ax.imshow(img[:, :, ::-1])
    c = cond[(s, q)]
    ax.set_title(f"{s} {q[3:13]}\n{c['largura']}x{c['altura']} | {c['pessoas_por_quadro_media']:.1f} pess./q | lum. {c['luminancia_media']:.0f}",
                 fontsize=7)
fig.suptitle("VisDrone2019-MOT: quadro central de cada sequência (val: 7; test-dev: 17)", fontsize=11)
fig.tight_layout()
os.makedirs("/w/validacao_videos/evidencias", exist_ok=True)
fig.savefig("/w/validacao_videos/evidencias/v00_mosaico_sequencias.png", dpi=100)
print("FIM_MOSAICO", flush=True)
