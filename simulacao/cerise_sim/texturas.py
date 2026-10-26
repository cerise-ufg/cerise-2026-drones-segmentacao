"""Texturas procedurais LADRILHAVEIS (sem imagens de terceiros / satelite): grama, terra, concreto,
asfalto, telhado e parede, sintetizadas por ruido espectral 1/f^beta (periodico por construcao, via FFT)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def ruido_periodico(n: int, beta: float, rng, anis: float = 1.0) -> np.ndarray:
    """Campo gaussiano periodico n x n com espectro de potencia ~ 1/f^beta, normalizado em [0, 1]."""
    fx = np.fft.fftfreq(n)[:, None]
    fy = np.fft.fftfreq(n)[None, :] * anis
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    amp = f ** (-beta / 2.0)
    amp[0, 0] = 0.0
    fase = np.exp(2j * np.pi * rng.random((n, n)))
    campo = np.real(np.fft.ifft2(amp * fase))
    campo -= campo.min()
    campo /= max(campo.max(), 1e-9)
    return campo


def _paleta(t: np.ndarray, cores) -> np.ndarray:
    cores = np.asarray(cores, float)
    pos = np.linspace(0, 1, len(cores))
    out = np.stack([np.interp(t, pos, cores[:, c]) for c in range(3)], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


def gerar(nome: str, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if nome in ("grama", "grama_seca"):
        macro = ruido_periodico(n, 2.6, rng)
        fino = ruido_periodico(n, 0.9, rng)
        laminas = ruido_periodico(n, 1.2, rng, anis=4.0)
        t = 0.55 * macro + 0.30 * fino + 0.15 * laminas
        if nome == "grama":
            cores = [(48, 64, 26), (72, 96, 38), (98, 122, 52), (128, 140, 70), (150, 146, 92)]
        else:
            cores = [(96, 88, 52), (128, 118, 70), (156, 144, 90), (178, 162, 110), (120, 128, 66)]
        img = _paleta(t, cores)
        # manchas de solo exposto
        manchas = ruido_periodico(n, 3.0, rng) > 0.82
        img[manchas] = (0.6 * img[manchas] + 0.4 * np.array([140, 112, 80])).astype(np.uint8)
        return img
    if nome == "terra":
        t = 0.6 * ruido_periodico(n, 2.2, rng) + 0.4 * ruido_periodico(n, 0.6, rng)
        return _paleta(t, [(110, 84, 60), (140, 110, 80), (168, 136, 100), (188, 160, 124)])
    if nome == "concreto":
        t = 0.5 * ruido_periodico(n, 2.0, rng) + 0.5 * ruido_periodico(n, 0.4, rng)
        img = _paleta(t, [(150, 148, 142), (172, 170, 164), (192, 190, 184), (208, 206, 200)])
        # juntas de dilatacao a cada 1/4 do ladrilho
        for k in range(0, n, n // 4):
            img[max(0, k - 1):k + 1, :] = (img[max(0, k - 1):k + 1, :] * 0.7).astype(np.uint8)
            img[:, max(0, k - 1):k + 1] = (img[:, max(0, k - 1):k + 1] * 0.7).astype(np.uint8)
        return img
    if nome == "asfalto":
        t = 0.4 * ruido_periodico(n, 1.8, rng) + 0.6 * ruido_periodico(n, 0.2, rng)
        return _paleta(t, [(46, 46, 48), (62, 62, 64), (78, 78, 80), (96, 96, 98)])
    if nome == "telhado":
        t = 0.7 * ruido_periodico(n, 1.6, rng) + 0.3 * ruido_periodico(n, 0.3, rng)
        img = _paleta(t, [(120, 122, 124), (150, 152, 154), (176, 178, 180)])
        for k in range(0, n, n // 16):  # ondulacao de telha metalica
            img[:, k:k + max(1, n // 64)] = (img[:, k:k + max(1, n // 64)] * 0.82).astype(np.uint8)
        return img
    if nome == "parede":
        t = 0.6 * ruido_periodico(n, 1.8, rng) + 0.4 * ruido_periodico(n, 0.3, rng)
        return _paleta(t, [(196, 186, 168), (214, 204, 186), (228, 220, 204)])
    raise ValueError(nome)


# nome -> (pixels, metros por ladrilho, rugosidade)
CATALOGO = {
    "grama": (2048, 24.0, 0.85),
    "grama_seca": (1024, 16.0, 0.9),
    "terra": (1024, 8.0, 0.9),
    "concreto": (1024, 8.0, 0.7),
    "asfalto": (1024, 8.0, 0.75),
    "telhado": (1024, 8.0, 0.5),
    "parede": (1024, 6.0, 0.8),
}


def salvar_todas(pasta, seed: int = 0, qualidade: int = 90) -> dict:
    from PIL import Image
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    out = {}
    for i, (nome, (n, metros, rug)) in enumerate(CATALOGO.items()):
        arq = pasta / f"{nome}.jpg"
        Image.fromarray(gerar(nome, n, seed + i)).save(arq, quality=qualidade)
        out[nome] = {"arquivo": f"texturas/{nome}.jpg", "pixels": n, "metros_por_ladrilho": metros, "rugosidade": rug}
    return out
