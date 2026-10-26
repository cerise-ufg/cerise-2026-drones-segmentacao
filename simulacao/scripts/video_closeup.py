"""Monta os quadros do clipe do close-up: camera do cinegrafista (1920x1080) com a camera do drone (gimbal, com as
caixas do YOLO e do gabarito) em miniatura e legendas. Depois: scripts/montar_video.sh (ffmpeg/libx264).

  python scripts/video_closeup.py --execucao saidas/closeup01 --saida tmp/video_closeup
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

FONTE = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")


def caixa_texto(dr, xy, txt, fonte, pad=8, fundo=(0, 0, 0, 150), cor=(255, 255, 255, 255)):
    x, y = xy
    l, t, r, b = dr.textbbox((x, y), txt, font=fonte)
    dr.rectangle((l - pad, t - pad, r + pad, b + pad), fill=fundo)
    dr.text((x, y), txt, font=fonte, fill=cor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execucao", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--largura_mini", type=int, default=512)
    ap.add_argument("--t_fim_mini", type=float, default=13.6,
                    help="depois disso a camera do drone olha para tras, atraves do trem de pouso: a miniatura some (0,4 s)")
    a = ap.parse_args()
    ex = Path(a.execucao)
    out = Path(a.saida)
    out.mkdir(parents=True, exist_ok=True)
    linhas = list(csv.DictReader(open(ex / "quadros.csv", encoding="utf-8")))
    f1 = ImageFont.truetype(FONTE, 26)
    f2 = ImageFont.truetype(FONTE, 22)
    f3 = ImageFont.truetype(FONTE, 17)

    def v(x, nd=1):
        return f"{x:.{nd}f}".replace(".", ",")
    for ln in linhas:
        q = int(ln["quadro"])
        t = float(ln["t"])
        base = Image.open(ex / "cinegrafista" / "rgb" / f"{q:06d}.jpg").convert("RGB")
        W, H = base.size
        cam = base.convert("RGBA")
        sob = Image.new("RGBA", cam.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(sob)
        caixa_texto(dr, (28, 26), "Isaac Sim 5.1 (RTX em tempo real) · pessoas animadas (omni.anim.people) · quadricóptero em voo baixo", f1)
        incl = float(ln["inclinacao"])
        caixa_texto(dr, (28, H - 58), f"t = {v(t)} s   câmera a {v(float(ln['cz']))} m de altura, {incl:.0f}° para baixo, "
                    f"campo de visão de {float(ln['hfov']):.0f}°", f2)
        cam.alpha_composite(sob)
        gim = ex / "gimbal" / "anotado" / f"{q:06d}.jpg"
        if not gim.exists():
            gim = ex / "gimbal" / "rgb" / f"{q:06d}.jpg"
        alfa = min(max((a.t_fim_mini + 0.4 - t) / 0.4, 0.0), 1.0)
        if gim.exists() and alfa > 0:
            g = Image.open(gim).convert("RGB")
            wm = a.largura_mini
            hm = int(round(g.size[1] * wm / g.size[0]))
            g = g.resize((wm, hm), Image.LANCZOS)
            x0, y0 = W - wm - 28, H - hm - 28
            camada = Image.new("RGBA", cam.size, (0, 0, 0, 0))
            dr = ImageDraw.Draw(camada)
            dr.rectangle((x0 - 3, y0 - 3, x0 + wm + 2, y0 + hm + 2), fill=(255, 255, 255, 255))
            camada.paste(g, (x0, y0))
            d = ((float(ln["gcx"]) - float(ln["centro_x"])) ** 2 + (float(ln["gcy"]) - float(ln["centro_y"])) ** 2
                 + (float(ln["gcz"]) - 0.9) ** 2) ** 0.5
            caixa_texto(dr, (x0 + 4, y0 - 64), f"câmera do drone: {d:.0f} m do grupo", f3, pad=5)
            caixa_texto(dr, (x0 + 4, y0 - 36), "YOLO11s (vermelho) · gabarito (verde)", f3, pad=5)
            if alfa < 1:
                camada.putalpha(camada.getchannel("A").point(lambda p: int(p * alfa)))
            cam.alpha_composite(camada)
        cam.convert("RGB").save(out / f"{q:06d}.jpg", quality=94, subsampling=0)
    print(f"{len(linhas)} quadros em {out}")


if __name__ == "__main__":
    main()
