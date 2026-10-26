"""Figuras do close-up (a partir dos quadros de alta qualidade gravados por isaac/closeup_isaac.py).

  sim_19_closeup_pessoas_drone.png : (a) grupo de perto, (b) passagem do drone, (c) detalhe do drone, (d) detalhe das pessoas
  sim_20_closeup_camera_drone.png  : camera do drone (gimbal) com caixas do YOLO11s (vermelho) e do gabarito (verde)

  python scripts/figuras_closeup.py --execucao <copia local de saidas/closeup01> --yolo resultados/closeup_yolo_gimbal.json \
      --saida evidencias
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cerise_sim.camera import Intrinsecos, Pose, olhar_para, projetar  # noqa: E402


def virgula(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",")


def ler_quadros(ex):
    return {round(float(r["t"]), 2): r for r in csv.DictReader(open(ex / "quadros.csv", encoding="utf-8"))}


def projetar_px(r, P, largura, altura):
    C = np.array([float(r["cx"]), float(r["cy"]), float(r["cz"])])
    alvo = np.array([float(r["alvo_x"]), float(r["alvo_y"]), float(r["alvo_z"])])
    K = Intrinsecos.de_hfov(largura, altura, float(r["hfov"]))
    uv, _z = projetar(np.atleast_2d(P), Pose(C, olhar_para(C, alvo)), K)
    return uv


def recorte(img, centro, w, h):
    W, H = img.size
    x0 = int(np.clip(centro[0] - w / 2, 0, W - w)); y0 = int(np.clip(centro[1] - h / 2, 0, H - h))
    return img.crop((x0, y0, x0 + w, y0 + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execucao", required=True)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--t_grupo", type=float, default=4.5)
    ap.add_argument("--t_drone", type=float, default=10.0)
    ap.add_argument("--t_gimbal", default="6.5,15.5")
    a = ap.parse_args()
    ex = Path(a.execucao)
    out = Path(a.saida)
    Q = ler_quadros(ex)
    meta = json.loads((ex / "meta.json").read_text(encoding="utf-8"))
    ya = json.loads(Path(a.yolo).read_text(encoding="utf-8"))
    pq = {d["quadro"]: d for d in ya["por_quadro"]}

    ra, rb = Q[round(a.t_grupo, 2)], Q[round(a.t_drone, 2)]
    ia = Image.open(ex / "heroi" / f"cinegrafista_t{a.t_grupo:05.2f}.png").convert("RGB")
    ib = Image.open(ex / "heroi" / f"cinegrafista_t{a.t_drone:05.2f}.png").convert("RGB")
    W, H = ia.size
    # detalhe do drone em (b): projecao da posicao gravada; detalhe do grupo em (a): centroide do grupo a 0,9 m
    uv_d = projetar_px(rb, [float(rb["dx"]), float(rb["dy"]), float(rb["dz"])], W, H)[0]
    uv_g = projetar_px(ra, [float(ra["centro_x"]), float(ra["centro_y"]), 0.9], W, H)[0]
    zd = recorte(ib, uv_d + np.array([0, 12]), 384, 216).resize((1152, 648), Image.LANCZOS)
    zg = recorte(ia, uv_g + np.array([0, 40]), 768, 432).resize((1152, 648), Image.LANCZOS)

    def dist(r):
        return float(np.hypot(float(r["cx"]) - float(r["centro_x"]), float(r["cy"]) - float(r["centro_y"])))
    fig, axs = plt.subplots(2, 2, figsize=(16, 10.3))
    paineis = [
        (ia, f"(a) t = {virgula(a.t_grupo)} s: grupo de 3 pessoas caminhando; câmera a {virgula(float(ra['cz']))} m de altura,\n"
             f"{virgula(dist(ra))} m do grupo, {virgula(float(ra['inclinacao']), 0)}° para baixo, campo de visão de {virgula(float(ra['hfov']), 0)}°"),
        (ib, f"(b) t = {virgula(a.t_drone)} s: quadricóptero a {virgula(float(rb['dz']))} m de altura passando sobre o grupo;\n"
             f"a câmera recua para {virgula(dist(rb))} m e levanta o enquadramento ({virgula(float(rb['inclinacao']), 0)}° para baixo, {virgula(float(rb['hfov']), 0)}°)"),
        (zd, "(c) detalhe de (b), ampliado 3×: quadricóptero com 0,9 m entre motores,\nhélices de 21\" girando, trem de pouso, gimbal e LEDs de navegação"),
        (zg, "(d) detalhe de (a), ampliado 1,5×: personagens do omni.anim.people em\nmeio passo, com sombras do sol (DistantLight) traçadas pelo RTX"),
    ]
    for ax, (im, tit) in zip(axs.flat, paineis):
        ax.imshow(im)
        ax.set_title(tit, fontsize=11.5, loc="left")
        ax.axis("off")
    fig.tight_layout(h_pad=2.6, w_pad=0.8)
    f19 = out / "sim_19_closeup_pessoas_drone.png"
    fig.savefig(f19, dpi=110)
    plt.close(fig)

    tg = [float(x) for x in a.t_gimbal.split(",")][:3]
    fig, axs = plt.subplots(2, 2, figsize=(16, 10.5))
    for k, (ax, t) in enumerate(zip(axs.flat, tg)):
        r = Q[round(t, 2)]
        q = int(r["quadro"])
        im = Image.open(ex / "gimbal" / "anotado" / f"{q:06d}.jpg").convert("RGB")
        d = float(np.linalg.norm(np.array([float(r["gcx"]) - float(r["centro_x"]), float(r["gcy"]) - float(r["centro_y"]),
                                           float(r["gcz"]) - 0.9])))
        incl = float(np.degrees(np.arctan2(float(r["gcz"]) - 0.9, np.hypot(float(r["gcx"]) - float(r["centro_x"]),
                                                                              float(r["gcy"]) - float(r["centro_y"])))))
        info = pq.get(q, {})
        ax.imshow(im)
        ax.set_title(f"({'abc'[k]}) t = {virgula(t)} s: drone a {virgula(d)} m do grupo, câmera {virgula(incl, 0)}° abaixo da horizontal\n"
                     f"gabarito: {info.get('gt', '?')} pessoas; YOLO11s (conf ≥ {virgula(ya['limiar'], 2)}): {info.get('det', '?')} caixas, "
                     f"{info.get('tp', '?')} corretas (IoU ≥ 0,5)", fontsize=11.5, loc="left")
        ax.axis("off")
    ax = axs.flat[3]
    fx = ya["por_faixa_distancia"]
    rot = [f"{f['dist_m'][0]}–{f['dist_m'][1]} m\n({virgula(f['inclinacao_graus'][0], 0)}–{virgula(f['inclinacao_graus'][1], 0)}°)\n"
           f"{f['quadros']} quadros" for f in fx]
    x = np.arange(len(fx))
    b1 = ax.bar(x - 0.19, [f["precisao"] for f in fx], 0.36, label="precisão", color="#c0392b")
    b2 = ax.bar(x + 0.19, [f["revocacao"] for f in fx], 0.36, label="revocação", color="#2e86c1")
    for bb in list(b2):
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.015, virgula(bb.get_height(), 2), ha="center", fontsize=10)
    for bb, f in zip(b1, fx):  # precisao com o numero de deteccoes (poucas na faixa mais proxima)
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.015, f"{virgula(bb.get_height(), 2)}\n({f['deteccoes']} det.)", ha="center", fontsize=9.5)
    ax.set_xticks(x, rot, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _p: virgula(v, 1)))
    ax.set_ylabel("IoU ≥ 0,5, conf ≥ " + virgula(ya["limiar"], 2))
    ax.set_title(f"(d) YOLO11s na câmera do drone por distância ao grupo (todos os {ya['quadros']} quadros;\n"
                 f"{ya['caixas_gabarito']} caixas de gabarito; AP50 geral = {virgula(ya['ap50'], 3)})", fontsize=11.5, loc="left")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(h_pad=2.6)
    f20 = out / "sim_20_closeup_camera_drone.png"
    fig.savefig(f20, dpi=110)
    plt.close(fig)
    print(f19, f20, "render:", round(meta["tempo_laco_s"] / 60, 1), "min")


if __name__ == "__main__":
    main()
