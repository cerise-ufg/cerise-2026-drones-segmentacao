"""Execucao sintetica SEM renderizacao: gera poses das cameras e caixas 2D de gabarito projetando cada
pessoa (cilindro de 0,25 m de raio x 1,70 m) nas cameras dos drones, no mesmo formato de `dados.py`.

Serve para (1) testar o pipeline de contagem de ponta a ponta com o DetectorOraculo antes de existir
render, (2) estimar o tamanho das pessoas em pixels e o volume de quadros de cada cenario. Nao ha
oclusao (oclusao = 0) nem imagem RGB.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .camera import Intrinsecos, pose_drone, projetar, quat_de_matriz
from .dados import CAB_CAIXAS, CAB_GAB, CAB_POSES, escrever_csv
from .pessoas import Pessoa
from .trajetoria import Trajetoria

_ANG = np.linspace(0, 2 * np.pi, 8, endpoint=False)


def caixa_pessoa(xy, pose, K: Intrinsecos, raio: float = 0.25, altura: float = 1.70):
    """Caixa 2D (x1,y1,x2,y2) recortada na imagem, ou None se fora do quadro/atras da camera."""
    circ = np.column_stack([xy[0] + raio * np.cos(_ANG), xy[1] + raio * np.sin(_ANG)])
    P = np.vstack([np.column_stack([circ, np.zeros(8)]), np.column_stack([circ, np.full(8, altura)])])
    uv, z = projetar(P, pose, K)
    if np.any(z <= 0.05):
        return None
    x1, y1 = uv.min(axis=0); x2, y2 = uv.max(axis=0)
    if x2 < 0 or y2 < 0 or x1 > K.largura or y1 > K.altura:
        return None
    xc1, yc1 = max(x1, 0.0), max(y1, 0.0)
    xc2, yc2 = min(x2, float(K.largura)), min(y2, float(K.altura))
    # descarta se menos de 30% da caixa ficou dentro do quadro
    if (xc2 - xc1) * (yc2 - yc1) < 0.3 * (x2 - x1) * (y2 - y1):
        return None
    return [xc1, yc1, xc2, yc2]


def simular_sem_render(cenario: dict, pessoas: list[Pessoa], fps: float, pasta=None, cams=None,
                       duracao_s: float | None = None) -> dict:
    dur = duracao_s or cenario["duracao_s"]
    tt = np.arange(0.0, dur + 1e-9, 1.0 / fps)
    meta = {"simulador": "sintetico_sem_render", "cenario": cenario["nome"], "fps": fps,
            "origem_enu": cenario["origem_enu"], "cameras": {}, "pessoas": [p.id for p in pessoas]}
    pos = {p.id: p.posicao(tt) for p in pessoas}
    gab = []
    for k, t in enumerate(tt):
        for p in pessoas:
            x, y, w = pos[p.id]
            gab.append((k, round(float(t), 4), p.id, round(float(x[k]), 3), round(float(y[k]), 3), 0.0, round(float(w[k]), 2)))
    saida = {"poses": {}, "caixas": {}, "gabarito": gab}
    for d in cenario["drones"]:
        if cams and d["id"] not in cams:
            continue
        K = Intrinsecos.de_dict(d["camera"]["intrinsecos"])
        incl = d["camera"].get("inclinacao_graus", 0.0)
        tr = Trajetoria.de_dict(d["trajetoria"])
        x, y, z, yaw, cobre, _ = tr.amostrar(tt)
        poses, caixas = [], []
        for k, t in enumerate(tt):
            pose = pose_drone(x[k], y[k], max(z[k], 0.5), yaw[k], incl)
            q = quat_de_matriz(pose.R_wc)
            poses.append((k, round(float(t), 4), *np.round(pose.C, 4), *np.round(q, 6), round(float(yaw[k]), 3), int(cobre[k])))
            if z[k] < 1.0:
                continue
            alcance = z[k] * (1.6 + 2.0 * np.tan(np.radians(incl))) + 5.0
            for p in pessoas:
                px, py, _ = pos[p.id]
                if abs(px[k] - x[k]) > alcance or abs(py[k] - y[k]) > alcance:
                    continue
                b = caixa_pessoa((px[k], py[k]), pose, K)
                if b is not None:
                    caixas.append((k, p.id, *np.round(b, 2), 0.0))
        meta["cameras"][d["id"]] = {"tipo": "drone", "intrinsecos": K.para_dict()}
        saida["poses"][d["id"]] = poses
        saida["caixas"][d["id"]] = caixas
    if pasta is not None:
        pasta = Path(pasta)
        pasta.mkdir(parents=True, exist_ok=True)
        (pasta / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        escrever_csv(pasta / "gabarito_pessoas.csv", CAB_GAB, gab)
        for cam in saida["poses"]:
            escrever_csv(pasta / cam / "poses.csv", CAB_POSES, saida["poses"][cam])
            escrever_csv(pasta / cam / "caixas_gt.csv", CAB_CAIXAS, saida["caixas"][cam])
    saida["meta"] = meta
    return saida
