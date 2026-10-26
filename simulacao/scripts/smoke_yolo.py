"""Teste rapido do detector do pipeline (DetectorYOLO, FP32) em imagens do VisDrone val ja presentes no Spark:
confere carga dos pesos ajustados, classes, ausencia de NaN e mede FPS no GB10 com imgsz 1280.
Uso (conteiner cerise26_drones_sim): python scripts/smoke_yolo.py --pesos /w/resultados/best_yolo11s_visdrone_pessoa_960.pt \
        --imagens /w/datasets/VisDrone/images/val --n 40 --saida resultados/smoke_yolo.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
import numpy as np  # noqa: E402

from cerise_sim.contagem import DetectorYOLO  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pesos", required=True)
ap.add_argument("--imagens", required=True)
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--saida", required=True)
a = ap.parse_args()
imgs = sorted(p for p in Path(a.imagens).rglob("*.jpg"))[: a.n]
det = DetectorYOLO(a.pesos, imgsz=a.imgsz)
det(imgs[0])  # aquecimento
import torch  # noqa: E402

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
n_cx, nan = [], 0
for p in imgs:
    b, s = det(p)
    n_cx.append(len(b))
    nan += int(np.isnan(b).any() or np.isnan(s).any())
dt = time.perf_counter() - t0
r = {"pesos": a.pesos, "classes_modelo": det.nomes, "filtro_classes": det.classes, "imgsz": a.imgsz, "fp32": True,
     "imagens": len(imgs), "fps": len(imgs) / dt, "deteccoes_por_imagem_media": float(np.mean(n_cx)),
     "imagens_com_nan": nan, "pico_memoria_gpu_GB": torch.cuda.max_memory_allocated() / 1e9,
     "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
Path(a.saida).write_text(json.dumps(r, indent=1), encoding="utf-8")
print(json.dumps(r, indent=1))
