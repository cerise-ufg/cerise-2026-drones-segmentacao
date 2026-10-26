"""Etapa 2 (CPU): pre-calcula, uma vez por sequencia, a compensacao de movimento de camera (GMC) que o BoT-SORT
do ultralytics aplica a cada quadro (metodo sparseOptFlow, downscale 2 -- o padrao do botsort.yaml), e mede
atributos de condicao por sequencia: luminancia media (noite), movimento de camera, resolucao, densidade e
escala das pessoas no gabarito (proxy de altitude).

A GMC nao depende das deteccoes nem dos parametros do rastreador, entao o resultado e reaproveitado por todas
as configuracoes do BoT-SORT em 30_rastrear.py (mesma sequencia de matrizes para todas -> comparacao justa).
Saida: cache_gmc/<split>/<seq>.npy (T x 2 x 3) e resultados/condicoes_sequencias.json
"""
import glob
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np
from ultralytics.trackers.utils.gmc import GMC

import metricas_mot as mm

MOT = "/w/mot"
OUT = "/w/validacao_videos/cache_gmc"
cv2.setNumThreads(1)


def processa(arg):
    split, sdir = arg
    seq = os.path.basename(sdir)
    arqs = sorted(glob.glob(f"{sdir}/*.jpg"))
    dest = f"{OUT}/{split}/{seq}.npy"
    gmc = GMC(method="sparseOptFlow")
    W, lum = [], []
    for i, a in enumerate(arqs):
        img = cv2.imread(a)
        if os.path.exists(dest):
            if i % 10 == 0:
                lum.append(float(cv2.cvtColor(cv2.resize(img, (480, 270)), cv2.COLOR_BGR2GRAY).mean()))
            continue
        W.append(gmc.apply(img))
        if i % 10 == 0:
            lum.append(float(cv2.cvtColor(cv2.resize(img, (480, 270)), cv2.COLOR_BGR2GRAY).mean()))
    if not os.path.exists(dest):
        np.save(dest, np.asarray(W, np.float64))
    W = np.load(dest)
    h, w = img.shape[:2]
    gt, ign = mm.ler_gabarito(f"{MOT}/VisDrone2019-MOT-{split}/annotations/{seq}.txt", len(arqs))
    cont, alturas, ids = [], [], set()
    for t in range(1, len(arqs) + 1):
        gi, gb, _ = gt[t]
        gi, gb = mm.filtrar_ignorado(gi, gb, ign[t])
        cont.append(len(gi)); ids.update(gi.tolist())
        alturas.extend((gb[:, 3] - gb[:, 1]).tolist())
    desloc = np.hypot(W[1:, 0, 2], W[1:, 1, 2]) if len(W) > 1 else np.zeros(1)
    escala = np.sqrt(np.abs(np.linalg.det(W[1:, :, :2]))) if len(W) > 1 else np.ones(1)
    r = {"split": split, "seq": seq, "quadros": len(arqs), "largura": w, "altura": h,
         "luminancia_media": float(np.mean(lum)),
         "mov_camera_px_por_quadro_mediana": float(np.median(desloc)),
         "mov_camera_pct_largura_por_quadro_mediana": float(100 * np.median(desloc) / w),
         "mov_camera_pct_largura_por_quadro_p90": float(100 * np.percentile(desloc, 90) / w),
         "zoom_por_quadro_desvio": float(np.std(escala)),
         "pessoas_por_quadro_media": float(np.mean(cont)), "pessoas_por_quadro_max": int(np.max(cont)),
         "pessoas_unicas_gabarito": len(ids),
         "altura_pessoa_px_mediana": float(np.median(alturas)) if alturas else float("nan"),
         "altura_pessoa_pct_imagem_mediana": float(100 * np.median(alturas) / h) if alturas else float("nan")}
    print(json.dumps(r), flush=True)
    return r


if __name__ == "__main__":
    tarefas = []
    for split in ("val", "test-dev"):
        os.makedirs(f"{OUT}/{split}", exist_ok=True)
        tarefas += [(split, d) for d in sorted(glob.glob(f"{MOT}/VisDrone2019-MOT-{split}/sequences/*"))]
    tarefas.sort(key=lambda x: -len(os.listdir(x[1])))  # mais longas primeiro
    with Pool(int(os.environ.get("NPROC", "4"))) as p:
        res = p.map(processa, tarefas, chunksize=1)
    res.sort(key=lambda r: (r["split"] != "val", r["seq"]))
    json.dump(res, open("/w/validacao_videos/resultados/condicoes_sequencias.json", "w"), indent=1)
    print("FIM_GMC", flush=True)
