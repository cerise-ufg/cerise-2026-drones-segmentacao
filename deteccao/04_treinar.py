"""Fine-tuning curto do YOLO11 (pesos COCO) no treino do VisDrone2019-DET, classe unica 'pessoa'.

Uso:  python 04_treinar.py --modelo yolo11s.pt --imgsz 960 --batch 8 --horas 0.75 [--teste]
O parametro `time` do ultralytics limita o treino por tempo de relogio (o agendamento de LR e o
fechamento do mosaico sao recalculados para caber no tempo). Um limite de memoria por processo
(set_per_process_memory_fraction) garante o teto de GPU da frente no Spark (memoria unificada).
"""
import argparse
import json
import os
import time

import torch
from ultralytics import YOLO, settings

ap = argparse.ArgumentParser()
ap.add_argument("--modelo", default="yolo11s.pt")
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--horas", type=float, default=0.75)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--teto_gb", type=float, default=7.5)
ap.add_argument("--nome", default=None)
ap.add_argument("--teste", action="store_true", help="1 epoca numa fracao do treino (medir memoria/velocidade)")
ap.add_argument("--fracao", type=float, default=0.05)
ap.add_argument("--cache", default="False")
a = ap.parse_args()

settings.update({"datasets_dir": "/w/datasets", "weights_dir": "/w/weights", "runs_dir": "/w/runs"})
os.makedirs("/w/weights", exist_ok=True)
os.chdir("/w/weights")
total = torch.cuda.get_device_properties(0).total_memory
frac = min(1.0, a.teto_gb * 1e9 / total)
torch.cuda.set_per_process_memory_fraction(frac, 0)
print(f"GPU {torch.cuda.get_device_name(0)} total {total/1e9:.1f} GB -> fracao {frac:.4f} (teto {a.teto_gb} GB)", flush=True)

nome = a.nome or f"ft_pessoa_{a.modelo[:-3]}_{a.imgsz}" + ("_teste" if a.teste else "")
model = YOLO(a.modelo)
mem = {"pico": 0.0}


def cb_mem(trainer):
    mem["pico"] = max(mem["pico"], torch.cuda.max_memory_reserved() / 1e9)


model.add_callback("on_train_batch_end", cb_mem)
t0 = time.time()
kw = dict(data="/w/datasets/visdrone_pessoa/visdrone_pessoa.yaml", imgsz=a.imgsz, batch=a.batch,
          workers=a.workers, device=0, project="/w/runs", name=nome, exist_ok=True, amp=False,  # FP16/autocast gera NaN no GB10 (ver teste_amp.py)
         
          cache=(False if a.cache == "False" else a.cache), plots=True, seed=0, deterministic=False, patience=100, cos_lr=True)
if a.teste:
    kw.update(epochs=1, fraction=a.fracao, val=False)
else:
    kw.update(epochs=a.epochs, time=a.horas)
model.train(**kw)
dt = time.time() - t0
out = {"modelo": a.modelo, "imgsz": a.imgsz, "batch": a.batch, "horas_limite": a.horas,
       "tempo_treino_s": dt, "pico_mem_gpu_reservada_GB": mem["pico"], "teto_GB": a.teto_gb, "nome": nome}
print(json.dumps(out), flush=True)
os.makedirs("/w/resultados", exist_ok=True)
json.dump(out, open(f"/w/resultados/treino_{nome}.json", "w"), indent=1)
