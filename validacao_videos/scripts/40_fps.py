"""Etapa 4 (GPU): vazao do pipeline completo no GB10, em fluxo (lote 1, quadro a quadro, FP32), como num drone:
leitura/decodificacao do JPEG -> deteccao (YOLO11s) -> rastreador (ByteTrack ou BoT-SORT com GMC real sobre o quadro).
Mede cada etapa separadamente e o FPS de ponta a ponta, em uma sequencia 4K e uma 1344x756 (300 quadros cada,
10 de aquecimento). A carga de CPU do Spark no momento e registrada (o Spark esta compartilhado).
Saida: resultados/fps_gb10.json
"""
import glob
import json
import os
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO, settings
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

settings.update({"datasets_dir": "/w/datasets", "weights_dir": "/w/weights", "runs_dir": "/w/runs"})
torch.cuda.set_per_process_memory_fraction(min(1.0, 7.5e9 / torch.cuda.get_device_properties(0).total_memory), 0)
FT = "/w/runs/ft_pessoa_yolo11s_960/weights/best.pt"
sel = json.load(open("/w/validacao_videos/resultados/selecao_val.json"))
SEQS = {"4K (3840x2160)": "/w/mot/VisDrone2019-MOT-val/sequences/uav0000268_05773_v",
        "1344x756": "/w/mot/VisDrone2019-MOT-val/sequences/uav0000086_00000_v"}
CASOS = [("ft960", FT, 960, None), ("ft1280", FT, 1280, None), ("coco960", "/w/weights/yolo11s.pt", 960, [0])]
N, AQ = 300, 10
res = {"carga_cpu_inicio": os.getloadavg(), "medidas": []}
for rot, d in SEQS.items():
    arqs = sorted(glob.glob(f"{d}/*.jpg"))[:N + AQ]
    for var, w, sz, cls in CASOS:
        model = YOLO(w)
        for nome in ("bytetrack", "botsort"):
            e = sel["escolha"][f"{var}|{nome}"]["cfg"]
            args = IterableSimpleNamespace(**{**YAML.load(check_yaml(f"{nome}.yaml")), **e})
            trk = BOTSORT(args) if nome == "botsort" else BYTETracker(args)
            t_le, t_det, t_trk, t_tot = [], [], [], []
            torch.cuda.reset_peak_memory_stats()
            for i, a in enumerate(arqs):
                t0 = time.perf_counter()
                img = cv2.imread(a)
                t1 = time.perf_counter()
                r = model.predict(img, imgsz=sz, conf=0.01, iou=0.6, max_det=1000, half=False, classes=cls, device=0, verbose=False)[0]
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                trk.update(r.boxes.cpu().numpy(), img)
                t3 = time.perf_counter()
                if i >= AQ:
                    t_le.append(t1 - t0); t_det.append(t2 - t1); t_trk.append(t3 - t2); t_tot.append(t3 - t0)
            m = {"sequencia": rot, "detector": var, "imgsz": sz, "rastreador": nome, "quadros": len(t_tot),
                 "ms_leitura_jpeg": 1000 * float(np.mean(t_le)), "ms_deteccao": 1000 * float(np.mean(t_det)),
                 "ms_rastreador": 1000 * float(np.mean(t_trk)), "ms_total": 1000 * float(np.mean(t_tot)),
                 "fps_ponta_a_ponta": 1.0 / float(np.mean(t_tot)), "fps_sem_leitura": 1.0 / float(np.mean(np.add(t_det, t_trk))),
                 "fps_so_deteccao": 1.0 / float(np.mean(t_det)), "mem_gpu_pico_GB": torch.cuda.max_memory_allocated() / 1e9}
            res["medidas"].append(m)
            print(json.dumps(m), flush=True)
res["carga_cpu_fim"] = os.getloadavg()
res["gpu"] = torch.cuda.get_device_name(0)
res["torch"] = torch.__version__
json.dump(res, open("/w/validacao_videos/resultados/fps_gb10.json", "w"), indent=1)
print("FIM_FPS", flush=True)
