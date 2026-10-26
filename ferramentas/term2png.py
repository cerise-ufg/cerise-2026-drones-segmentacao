#!/usr/bin/env python3
"""Renderiza texto de terminal (stdin ou arquivo) como um "print" PNG estilo janela.

Uso:
    comando | python term2png.py -o saida.png -t "titulo da janela"
    python term2png.py -i log.txt -o saida.png -t "spark-a55b: treino" --prompt "$ python train.py"

Depende só de Pillow. Remove códigos ANSI. Corta em --max-lines (mantém o fim).
"""
import argparse
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

FONT_CANDIDATES = [
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/cour.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]


def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", help="arquivo de texto (padrão: stdin)")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-t", "--title", default="terminal")
    ap.add_argument("--prompt", default=None, help="linha de comando mostrada no topo")
    ap.add_argument("--max-lines", type=int, default=60)
    ap.add_argument("--width-chars", type=int, default=120)
    ap.add_argument("--font-size", type=int, default=16)
    args = ap.parse_args()

    raw = open(args.input, encoding="utf-8", errors="replace").read() if args.input else sys.stdin.read()
    lines = ANSI.sub("", raw).replace("\t", "    ").splitlines()
    if args.prompt:
        lines = [args.prompt] + lines
    if len(lines) > args.max_lines:
        lines = lines[: 1] + ["..."] + lines[-(args.max_lines - 2):] if args.prompt else ["..."] + lines[-(args.max_lines - 1):]
    wrapped = []
    for ln in lines:
        while len(ln) > args.width_chars:
            wrapped.append(ln[: args.width_chars])
            ln = "  " + ln[args.width_chars:]
        wrapped.append(ln)

    font = load_font(args.font_size)
    bbox = font.getbbox("M")
    cw, ch = bbox[2] - bbox[0], int((bbox[3] - bbox[1]) * 1.55) + 2
    pad, bar = 18, 34
    w = pad * 2 + cw * min(args.width_chars, max((len(l) for l in wrapped), default=20))
    w = max(w, 640)
    h = bar + pad * 2 + ch * len(wrapped)

    img = Image.new("RGB", (w, h), (30, 30, 36))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, bar], fill=(52, 52, 60))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([14 + i * 22, 11, 26 + i * 22, 23], fill=c)
    tf = load_font(max(12, args.font_size - 2))
    tw = d.textlength(args.title, font=tf)
    d.text(((w - tw) / 2, 9), args.title, fill=(210, 210, 215), font=tf)

    y = bar + pad
    for idx, ln in enumerate(wrapped):
        color = (120, 220, 140) if (args.prompt and idx == 0) else (225, 225, 230)
        if re.search(r"error|erro|failed|traceback", ln, re.I):
            color = (255, 120, 110)
        d.text((pad, y), ln, fill=color, font=font)
        y += ch
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    img.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
