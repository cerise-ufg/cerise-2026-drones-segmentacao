"""Avalia o YOLO11 pre-treinado no COCO (classe 0 = person) no VisDrone2019-DET val, classe pessoa.

As predicoes sao filtradas para a classe COCO 0 (classes=[0]), que coincide com a classe 0 'pessoa'
do subconjunto preparado. Metricas do validador do ultralytics (mAP50, mAP50-95, P, R).
"""
import json
import os
import sys
import time

import torch
from ultralytics import YOLO, settings

settings.update({"datasets_dir": "/w/datasets", "weights_dir": "/w/weights", "runs_dir": "/w/runs"})
os.makedirs("/w/weights", exist_ok=True)
os.chdir("/w/weights")  # pesos oficiais baixados aqui (github.com/ultralytics/assets)
DATA = "/w/datasets/visdrone_pessoa/visdrone_pessoa.yaml"
CONFIGS = [("yolo11n.pt", 640), ("yolo11n.pt", 960), ("yolo11s.pt", 640), ("yolo11s.pt", 960),
           ("yolo11m.pt", 640), ("yolo11m.pt", 960), ("yolo11s.pt", 1280)]
if len(sys.argv) > 1:
    CONFIGS = [(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[1:]]

res = []
for w, imgsz in CONFIGS:
    torch.cuda.reset_peak_memory_stats()
    model = YOLO(w)
    t0 = time.time()
    m = model.val(data=DATA, imgsz=imgsz, batch=8, classes=[0], conf=0.001, iou=0.7, half=False,  # FP16 gera NaN no GB10 (teste_amp.py)
                 
                  device=0, workers=4, plots=False, verbose=False, project="/w/runs/val",
                  name=f"coco_{w[:-3]}_{imgsz}", exist_ok=True)
    r = {"modelo": w, "pesos": "COCO pre-treinado", "imgsz": imgsz, "split": "val",
         "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map), "precisao": float(m.box.mp),
         "revocacao": float(m.box.mr), "velocidade_ms": {k: float(v) for k, v in m.speed.items()},
         "pico_mem_gpu_GB": torch.cuda.max_memory_reserved() / 1e9, "tempo_total_s": time.time() - t0}
    print(json.dumps(r), flush=True)
    res.append(r)
    json.dump(res, open("/w/resultados/val_coco_pretreinado.json", "w"), indent=1)
