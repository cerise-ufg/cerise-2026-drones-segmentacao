"""Avaliacao do modelo ajustado (fine-tuning) x pre-treinado COCO: mAP (val e test-dev), erro de contagem
por imagem (MAE/MAPE), FPS no GB10 e imagens de exemplo.

Protocolo de contagem (sem vazamento): o limiar de confianca e escolhido no VAL (minimo MAE) e aplicado
ao TEST-DEV. Contagem 'mascarada' ignora deteccoes cujo centro cai nas regioes ignoradas do VisDrone
(categoria 0: multidoes/areas nao anotadas), como no protocolo oficial do VisDrone.
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO, settings

settings.update({"datasets_dir": "/w/datasets", "weights_dir": "/w/weights", "runs_dir": "/w/runs"})
os.chdir("/w/weights")
torch.cuda.set_per_process_memory_fraction(min(1.0, 7.5e9 / torch.cuda.get_device_properties(0).total_memory), 0)
D = "/w/datasets/visdrone_pessoa"
DATA = f"{D}/visdrone_pessoa.yaml"
FT = sys.argv[1] if len(sys.argv) > 1 else "/w/runs/ft_pessoa_yolo11s_960/weights/best.pt"
OUT = "/w/resultados"
os.makedirs(f"{OUT}/exemplos", exist_ok=True)
IMGSZ = 960
MODELOS = {"yolo11s COCO (pre-treinado)": ("yolo11s.pt", [0]), "yolo11s ajustado VisDrone-pessoa": (FT, None)}
res = {"modelo_ft": FT, "imgsz": IMGSZ, "metricas": [], "contagem": {}, "fps": []}


def salvar():
    json.dump(res, open(f"{OUT}/avaliacao_deteccao.json", "w"), indent=1)


# 1) mAP (validador oficial do ultralytics, FP32)
for nome, (w, cls) in MODELOS.items():
    for split in ("val", "test"):
        m = YOLO(w).val(data=DATA, split=split, imgsz=IMGSZ, batch=8, half=False, conf=0.001, iou=0.7, max_det=1000,
                        classes=cls, device=0, workers=4, plots=(split == "val"), verbose=False,
                        project="/w/runs/val", name=f"{'ft' if cls is None else 'coco'}_{split}_{IMGSZ}", exist_ok=True)
        r = {"modelo": nome, "split": split, "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
             "precisao": float(m.box.mp), "revocacao": float(m.box.mr), "velocidade_ms": {k: float(v) for k, v in m.speed.items()}}
        print(json.dumps(r), flush=True)
        res["metricas"].append(r)
        salvar()


# 2) contagem por imagem
def predizer(w, cls, arquivos):
    model = YOLO(w)
    out = {}
    for i in range(0, len(arquivos), 8):
        lote = arquivos[i:i + 8]
        rs = model.predict(lote, imgsz=IMGSZ, conf=0.05, iou=0.6, max_det=1000, half=False, classes=cls,
                           device=0, verbose=False)
        for f, r in zip(lote, rs):
            b = r.boxes
            out[os.path.basename(f)] = (b.xyxy.cpu().numpy(), b.conf.cpu().numpy())
    return out


def centro_ignorado(xyxy, ign):
    if len(ign) == 0 or len(xyxy) == 0:
        return np.zeros(len(xyxy), dtype=bool)
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2; cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
    m = np.zeros(len(xyxy), dtype=bool)
    for x, y, w, h in ign:
        m |= (cx >= x) & (cx <= x + w) & (cy >= y) & (cy <= y + h)
    return m


def erros(pred, gt):
    pred = np.asarray(pred, float); gt = np.asarray(gt, float)
    e = pred - gt
    nz = gt > 0
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "MAPE_pct": float(100 * np.mean(np.abs(e[nz]) / gt[nz])), "vies_medio": float(e.mean()),
            "erro_total_pct": float(100 * (pred.sum() / gt.sum() - 1)), "n_imagens": int(len(gt)),
            "gt_total": int(gt.sum()), "pred_total": int(pred.sum())}


LIMIARES = np.round(np.arange(0.05, 0.701, 0.025), 3)
for nome, (w, cls) in MODELOS.items():
    cont = {}
    preds = {}
    for split in ("val", "test"):
        gt = json.load(open(f"{D}/gt_{split}.json"))
        arqs = sorted(glob.glob(f"{D}/images/{split}/*.jpg"))
        preds[split] = predizer(w, cls, arqs)
        tab = []
        for thr in LIMIARES:
            p_raw, p_msk, g = [], [], []
            for f in arqs:
                k = os.path.basename(f)
                xyxy, conf = preds[split][k]
                sel = conf >= thr
                ign = centro_ignorado(xyxy[sel], gt[k]["ignoradas"])
                p_raw.append(int(sel.sum())); p_msk.append(int(sel.sum() - ign.sum())); g.append(gt[k]["pessoas"])
            tab.append({"limiar": float(thr), "bruta": erros(p_raw, g), "mascarada": erros(p_msk, g)})
        cont[split] = tab
    # limiar escolhido no val (minimo MAE da contagem mascarada) e aplicado ao test-dev
    i_best = int(np.argmin([t["mascarada"]["MAE"] for t in cont["val"]]))
    thr = float(LIMIARES[i_best])
    res["contagem"][nome] = {"limiar_escolhido_no_val": thr, "val": cont["val"][i_best], "test": cont["test"][i_best],
                             "test_conf_0.25": cont["test"][int(np.argmin(np.abs(LIMIARES - 0.25)))],
                             "curva_val": [{"limiar": t["limiar"], "MAE_mascarada": t["mascarada"]["MAE"], "MAE_bruta": t["bruta"]["MAE"]} for t in cont["val"]],
                             "curva_test": [{"limiar": t["limiar"], "MAE_mascarada": t["mascarada"]["MAE"], "MAE_bruta": t["bruta"]["MAE"]} for t in cont["test"]]}
    print(nome, "limiar", thr, "val", cont["val"][i_best]["mascarada"], "test", cont["test"][i_best]["mascarada"], flush=True)
    salvar()
    # dispersao contagem prevista x real (test-dev) para o grafico
    gt = json.load(open(f"{D}/gt_test.json"))
    disp = []
    for k, (xyxy, conf) in preds["test"].items():
        sel = conf >= thr
        ign = centro_ignorado(xyxy[sel], gt[k]["ignoradas"])
        disp.append([gt[k]["pessoas"], int(sel.sum() - ign.sum())])
    res["contagem"][nome]["dispersao_test"] = disp
    salvar()
    if cls is None:
        # 4) exemplos com deteccoes (modelo ajustado, limiar escolhido)
        gtv = json.load(open(f"{D}/gt_val.json"))
        ordem = sorted(gtv, key=lambda k: gtv[k]["pessoas"])
        escolha = [ordem[int(q * (len(ordem) - 1))] for q in (0.55, 0.8, 0.93, 0.99)]
        for k in escolha:
            im = cv2.imread(f"{D}/images/val/{k}")
            xyxy, conf = preds["val"][k]
            sel = conf >= thr
            ign = centro_ignorado(xyxy[sel], gtv[k]["ignoradas"])
            for (x, y, w_, h) in gtv[k]["ignoradas"]:
                cv2.rectangle(im, (x, y), (x + w_, y + h), (128, 128, 128), 2)
            for (x0, y0, x1, y1), ig in zip(xyxy[sel].astype(int), ign):
                cv2.rectangle(im, (x0, y0), (x1, y1), (128, 128, 128) if ig else (0, 200, 255), 2)
            txt = f"real: {gtv[k]['pessoas']}  detectadas: {int(sel.sum() - ign.sum())}  (limiar {thr:.2f})"
            cv2.rectangle(im, (0, 0), (16 + 19 * len(txt), 50), (0, 0, 0), -1)
            cv2.putText(im, txt, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
            esc = 1280 / im.shape[1]
            cv2.imwrite(f"{OUT}/exemplos/{k}", cv2.resize(im, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
        res["exemplos"] = escolha
        salvar()

# 3) FPS no GB10 (lote 1, FP32, imagens pre-carregadas na memoria; GPU compartilhada com outros jobs)
arqs = sorted(glob.glob(f"{D}/images/val/*.jpg"))[:150]
imgs = [cv2.imread(f) for f in arqs]
for nome, w in (("yolo11s ajustado", FT), ("yolo11n COCO", "yolo11n.pt"), ("yolo11m COCO", "yolo11m.pt")):
    model = YOLO(w)
    for imgsz in (640, 960):
        for im in imgs[:15]:
            model.predict(im, imgsz=imgsz, half=False, device=0, verbose=False, conf=0.25)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sp = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
        for im in imgs:
            r = model.predict(im, imgsz=imgsz, half=False, device=0, verbose=False, conf=0.25)[0]
            for kk in sp:
                sp[kk] += r.speed[kk]
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        f = {"modelo": nome, "imgsz": imgsz, "lote": 1, "precisao": "FP32", "n_imagens": len(imgs),
             "fps_ponta_a_ponta": len(imgs) / dt, "ms_por_imagem": 1000 * dt / len(imgs),
             **{f"{kk}_ms": v / len(imgs) for kk, v in sp.items()}}
        print(json.dumps(f), flush=True)
        res["fps"].append(f)
        salvar()
print("FIM", flush=True)
