"""Baixa o VisDrone2019-DET pelo VisDrone.yaml oficial do ultralytics (fontes: GitHub ultralytics/assets)
e converte para o formato YOLO. Os dados ficam em /w/datasets (fora do git)."""
from ultralytics import settings
settings.update({"datasets_dir": "/w/datasets", "weights_dir": "/w/weights", "runs_dir": "/w/runs"})
from ultralytics.data.utils import check_det_dataset
import os

d = check_det_dataset("VisDrone.yaml", autodownload=True)
print({k: d[k] for k in ("path", "train", "val", "test", "nc") if k in d})
print(d["names"])
for root, dirs, files in os.walk(d["path"]):
    depth = root[len(str(d["path"])):].count(os.sep)
    if depth <= 3:
        print(root, len(files), "arquivos")
