"""Diagnostico do check de AMP do ultralytics no GB10: compara saidas FP32 x autocast FP16/BF16."""
import os
import glob
import torch
from ultralytics import YOLO
from ultralytics.utils import checks

os.chdir("/w/weights")
img = sorted(glob.glob("/w/datasets/visdrone_pessoa/images/val/*.jpg"))[0]
for w in ["yolo11s.pt", "yolo26n.pt"]:
    m = YOLO(w)
    a = m.predict(img, device=0, half=False, verbose=False, conf=0.25)[0].boxes
    with torch.autocast("cuda", dtype=torch.float16):
        b = m.predict(img, device=0, half=False, verbose=False, conf=0.25)[0].boxes
    c = m.predict(img, device=0, half=True, verbose=False, conf=0.25)[0].boxes
    print(w, "fp32 n=", len(a), "autocast16 n=", len(b), "half n=", len(c))
    if len(a) and len(b):
        n = min(len(a), len(b))
        print("  conf fp32", a.conf[:5].tolist(), "\n  conf amp ", b.conf[:5].tolist())
# reproduz o check oficial
from ultralytics.nn.tasks import DetectionModel
mm = YOLO("yolo11s.pt").model.cuda()
try:
    print("check_amp(yolo11s) =", checks.check_amp(mm))
except Exception as e:
    print("erro check_amp", repr(e))
x = torch.randn(1, 3, 640, 640, device="cuda")
net = YOLO("yolo11s.pt").model.cuda().eval()
with torch.no_grad():
    y32 = net(x)[0]
    with torch.autocast("cuda", dtype=torch.float16):
        y16 = net(x)[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        yb = net(x)[0]
print("max abs diff fp16", (y32 - y16.float()).abs().max().item(), "bf16", (y32 - yb.float()).abs().max().item(),
      "nan16", torch.isnan(y16).any().item())
