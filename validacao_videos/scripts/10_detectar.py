"""Etapa 1 da validacao em video: deteccao quadro a quadro em todas as sequencias do VisDrone2019-MOT
(val e test-dev), com cache das deteccoes para o rastreamento offline.

Variantes de detector (FP32 -- FP16 da NaN no GB10 com o PyTorch 2.12 do NGC 26.05):
  ft960   YOLO11s ajustado no VisDrone-DET (classe unica "pessoa"), imgsz 960
  ft1280  o mesmo modelo, imgsz 1280
  coco960 YOLO11s pre-treinado no COCO, so a classe 0 (person), imgsz 960
  coco1280 idem, imgsz 1280
Cada quadro e decodificado uma unica vez (pool de threads) e passa pelas 4 variantes. Limiar de confianca
baixo (0,01) e NMS IoU 0,6: os limiares de contagem/rastreamento sao escolhidos depois, so no val.
Saida: /w/validacao_videos/cache_det/<variante>/<split>/<seq>.npz  (quadro, xyxy, conf)
Uso (no container cerise26_drones, ver run_vv.sh): python 10_detectar.py
"""
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from ultralytics import YOLO, settings

settings.update({"datasets_dir": "/w/datasets", "weights_dir": "/w/weights", "runs_dir": "/w/runs"})
torch.cuda.set_per_process_memory_fraction(min(1.0, 7.5e9 / torch.cuda.get_device_properties(0).total_memory), 0)
MOT = "/w/mot"
OUT = "/w/validacao_videos/cache_det"
FT = "/w/runs/ft_pessoa_yolo11s_960/weights/best.pt"
VARIANTES = {"ft960": (FT, 960, None), "ft1280": (FT, 1280, None),
             "coco960": ("/w/weights/yolo11s.pt", 960, [0]), "coco1280": ("/w/weights/yolo11s.pt", 1280, [0])}
CONF, IOU, BATCH = 0.01, 0.6, 8
modelos = {k: YOLO(w) for k, (w, _, _) in VARIANTES.items()}
pool = ThreadPoolExecutor(4)
log = {"conf": CONF, "iou_nms": IOU, "batch": BATCH, "half": False, "variantes": {k: [v[0], v[1], v[2]] for k, v in VARIANTES.items()},
       "sequencias": []}

for split in ("val", "test-dev"):
    for sdir in sorted(glob.glob(f"{MOT}/VisDrone2019-MOT-{split}/sequences/*")):
        seq = os.path.basename(sdir)
        if all(os.path.exists(f"{OUT}/{k}/{split}/{seq}.npz") for k in VARIANTES):
            print("ja existe", split, seq, flush=True); continue
        arqs = sorted(glob.glob(f"{sdir}/*.jpg"))
        acc = {k: ([], [], []) for k in VARIANTES}
        tempo = {k: 0.0 for k in VARIANTES}
        t0 = time.time()
        prox = pool.map(cv2.imread, arqs[:BATCH])
        for i in range(0, len(arqs), BATCH):
            imgs = list(prox)
            if i + BATCH < len(arqs):
                prox = pool.map(cv2.imread, arqs[i + BATCH:i + 2 * BATCH])
            for k, (w, sz, cls) in VARIANTES.items():
                torch.cuda.synchronize(); ti = time.time()
                rs = modelos[k].predict(imgs, imgsz=sz, conf=CONF, iou=IOU, max_det=1000, half=False, classes=cls,
                                        device=0, verbose=False)
                torch.cuda.synchronize(); tempo[k] += time.time() - ti
                for j, r in enumerate(rs):
                    b = r.boxes
                    n = len(b)
                    acc[k][0].append(np.full(n, i + j + 1, np.int32))  # quadros numerados a partir de 1 (como o gabarito)
                    acc[k][1].append(b.xyxy.cpu().numpy().astype(np.float32))
                    acc[k][2].append(b.conf.cpu().numpy().astype(np.float32))
        h, w_ = imgs[0].shape[:2]
        for k in VARIANTES:
            os.makedirs(f"{OUT}/{k}/{split}", exist_ok=True)
            np.savez_compressed(f"{OUT}/{k}/{split}/{seq}.npz", quadro=np.concatenate(acc[k][0]),
                                xyxy=np.concatenate(acc[k][1]).reshape(-1, 4), conf=np.concatenate(acc[k][2]),
                                n_quadros=len(arqs), largura=w_, altura=h)
        info = {"split": split, "seq": seq, "quadros": len(arqs), "resolucao": [w_, h], "tempo_total_s": time.time() - t0,
                "ms_por_quadro_modelo": {k: 1000 * tempo[k] / len(arqs) for k in VARIANTES},
                "mem_gpu_pico_GB": torch.cuda.max_memory_allocated() / 1e9}
        log["sequencias"].append(info)
        print(json.dumps(info), flush=True)
        json.dump(log, open("/w/validacao_videos/resultados/log_deteccao.json", "w"), indent=1)
print("FIM_DETECCAO", flush=True)
