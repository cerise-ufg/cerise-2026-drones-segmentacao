"""Remove o disco solar de um HDRI latlong (Radiance .hdr) para usa-lo como cupula de ceu junto com um sol
explicito (DistantLight), evitando sombras duplas com direcoes diferentes. Os HDRIs usados sao da Poly Haven
(CC0), hospedados no servidor oficial de ativos da NVIDIA.

  python scripts/ceu_sem_sol.py <entrada.hdr> <saida.hdr> [--json info.json]
"""
import argparse
import json
import math

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada")
    ap.add_argument("saida")
    ap.add_argument("--json", default=None)
    ap.add_argument("--fator", type=float, default=25.0, help="pixels acima de fator x mediana do ceu = sol")
    a = ap.parse_args()
    img = cv2.imread(a.entrada, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR).astype(np.float32)
    h, w = img.shape[:2]
    lum = 0.0722 * img[..., 0] + 0.7152 * img[..., 1] + 0.2126 * img[..., 2]
    ceu = lum[: h // 2]
    med = float(np.median(ceu))
    iy, ix = np.unravel_index(int(np.argmax(lum)), lum.shape)
    el = 90.0 - 180.0 * (iy + 0.5) / h
    az_img = 360.0 * (ix + 0.5) / w
    mask = (lum > a.fator * med).astype(np.uint8)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    n = int(mask.sum())
    # substitui a regiao do sol pela media de um anel ao redor (ceu proximo), sem o pico
    anel = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))) - mask
    cor = img[anel.astype(bool)].mean(axis=0) if anel.any() else np.array([med] * 3, np.float32)
    out = img.copy()
    out[mask.astype(bool)] = cor
    lum2 = 0.0722 * out[..., 0] + 0.7152 * out[..., 1] + 0.2126 * out[..., 2]
    # parcela da irradiancia (hemisferio superior, ponderada pelo cosseno) que estava no sol
    th = (np.arange(h // 2) + 0.5) / h * math.pi
    pes = (np.sin(th) * np.cos(th))[:, None]
    e_antes = float((lum[: h // 2] * pes).sum())
    e_depois = float((lum2[: h // 2] * pes).sum())
    cv2.imwrite(a.saida, out)
    info = {"entrada": a.entrada, "saida": a.saida, "tamanho": [w, h], "mediana_ceu": med, "pixel_sol": [int(ix), int(iy)],
            "sol_elevacao_graus": round(el, 1), "sol_coluna_graus": round(az_img, 1), "pixels_removidos": n,
            "lum_max_antes": float(lum.max()), "lum_max_depois": float(lum2.max()),
            "fracao_irradiancia_no_sol": round(1 - e_depois / e_antes, 3),
            # irradiancia horizontal do ceu sem o sol, em unidades do arquivo (x intensidade da DomeLight)
            "irradiancia_ceu_unid": e_depois * (math.pi / h) * (2 * math.pi / w),
            "irradiancia_sol_unid": (e_antes - e_depois) * (math.pi / h) * (2 * math.pi / w)}
    print(json.dumps(info, indent=1))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1)


if __name__ == "__main__":
    main()
