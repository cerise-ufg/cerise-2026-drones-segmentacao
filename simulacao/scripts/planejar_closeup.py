"""Planeja o close-up "cinegrafista" e confere o enquadramento por projecao pinhole ANTES do render no Isaac.

Usa as posicoes REAIS das pessoas gravadas no voo run01 (gabarito do simulador, 5 Hz) para prever onde o grupo
estara; a camera do render segue o centroide medido ao vivo (isaac/closeup_isaac.py), entao o plano so precisa
ser aproximado. Gera <saida>.json (plano) e <saida>.png (quadros de conferencia).

  python scripts/planejar_closeup.py --gabarito saidas/run01/gabarito_pessoas.csv --cenario cenario/reduzido \
      --grupo P022,P023,P024 --t_inicio 3 --t_fim 17 --saida closeup/plano_closeup
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cerise_sim.camera import Intrinsecos, Pose, olhar_para, projetar  # noqa: E402
from cerise_sim.closeup import PlanoCloseup, inclinacao_graus  # noqa: E402


def ler_gabarito(caminho):
    P = defaultdict(list)
    with open(caminho, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            P[r["pessoa_id"]].append((float(r["t"]), float(r["x"]), float(r["y"])))
    return {k: np.array(sorted(v)) for k, v in P.items()}


def pos(P, pid, t):
    a = P[pid]
    return np.array([np.interp(t, a[:, 0], a[:, 1]), np.interp(t, a[:, 0], a[:, 2])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gabarito", required=True)
    ap.add_argument("--cenario", required=True)
    ap.add_argument("--grupo", default="P022,P023,P024")
    ap.add_argument("--t_inicio", type=float, default=3.0)
    ap.add_argument("--t_fim", type=float, default=17.0)
    ap.add_argument("--raio_pessoas", type=float, default=60.0, help="inclui no render quem passa a menos disso do grupo")
    ap.add_argument("--hfov", type=float, default=52.0)
    ap.add_argument("--hfov_passagem", type=float, default=66.0)
    ap.add_argument("--azimute_ini", type=float, default=35.0)
    ap.add_argument("--azimute_fim", type=float, default=85.0)
    ap.add_argument("--raio_ini", type=float, default=9.5)
    ap.add_argument("--raio_fim", type=float, default=10.5)
    ap.add_argument("--altura_ini", type=float, default=5.2)
    ap.add_argument("--altura_fim", type=float, default=5.2)
    ap.add_argument("--raio_extra", type=float, default=5.0, help="recuo da camera durante a passagem do drone (m)")
    ap.add_argument("--altura_extra", type=float, default=0.0, help="variacao da altura da camera durante a passagem (m)")
    ap.add_argument("--drone_alt", type=float, default=8.5)
    ap.add_argument("--drone_alem", type=float, default=0.0, help="m alem do grupo, na direcao media de visada")
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()

    P = ler_gabarito(a.gabarito)
    grupo = a.grupo.split(",")
    ts = np.round(np.arange(0.0, a.t_fim + 2.01, 0.2), 3)
    cen = np.array([np.mean([pos(P, g, t) for g in grupo], axis=0) for t in ts])
    centroide = [[float(t), float(x), float(y)] for t, (x, y) in zip(ts, cen)]

    # pessoas que passam perto do grupo durante a janela (as demais nao interagem com ele em < 20 s)
    sel = sorted(pid for pid in P if min(np.linalg.norm(pos(P, pid, t) - cen[i]) for i, t in enumerate(ts)) < a.raio_pessoas)

    # drone: reta perpendicular a visada media, alem do grupo; desacelera perto do grupo ("passagem de inspecao")
    tm = 0.5 * (a.t_inicio + a.t_fim)
    cm = np.array([np.interp(tm, ts, cen[:, 0]), np.interp(tm, ts, cen[:, 1])])
    az_m = math.radians(0.5 * (a.azimute_ini + a.azimute_fim))
    d = -np.array([math.cos(az_m), math.sin(az_m)])          # visada media (horizontal)
    r = np.array([d[1], -d[0]])                              # direita da imagem
    B = cm + a.drone_alem * d
    h = a.drone_alt
    pontos = [[a.t_inicio - 1.0, *(B - 30 * r), h + 1.5],
              [a.t_inicio + 4.0, *(B - 13 * r), h + 0.5],
              [tm, *(B - 1.5 * r), h],
              [tm + 2.5, *(B + 4.0 * r), h],
              [a.t_fim + 1.0, *(B + 22 * r), h + 1.0]]
    plano = {
        "grupo": grupo, "pessoas": sel, "t_inicio": a.t_inicio, "t_fim": a.t_fim,
        "centroide_pred": centroide,
        "camera": {"hfov": a.hfov, "hfov_passagem": a.hfov_passagem, "largura": 1920, "altura": 1080, "azimute_ini": a.azimute_ini, "azimute_fim": a.azimute_fim,
                   "raio_ini": a.raio_ini, "raio_fim": a.raio_fim, "altura_ini": a.altura_ini, "altura_fim": a.altura_fim,
                   "alvo_z": 1.0, "tilt_ini": a.t_inicio + 3.0, "tilt_dur": 2.5, "tilt_volta": a.t_fim - 3.5,
                   "raio_extra": a.raio_extra, "altura_extra": a.altura_extra, "fracao_pes": 0.5, "suavizacao_s": 1.5},
        "drone": {"pontos": [[round(float(v), 3) for v in p] for p in pontos], "envergadura_m": 0.9,
                  "rotacao_helices_hz": 7.0},
        "gimbal": {"hfov": 60.0, "largura": 1280, "altura": 720},
        "fonte": f"posicoes reais do run01 ({Path(a.gabarito).name})",
    }
    pl = PlanoCloseup(plano)
    Kg = Intrinsecos.de_hfov(1280, 720, 60.0)
    linhas, amostras = [], []
    fps = 24
    n_dr = 0
    for k, t in enumerate(np.arange(a.t_inicio, a.t_fim + 1e-6, 1.0 / fps)):
        c = np.array([np.interp(t, ts, cen[:, 0]), np.interp(t, ts, cen[:, 1])])
        C, alvo = pl.camera(t, c)
        Kc = Intrinsecos.de_hfov(1920, 1080, pl.hfov(t))
        pose = Pose(C, olhar_para(C, alvo))
        dpos, rumo, incl, vh = pl.drone_estado(t)
        uv, z = projetar(dpos[None], pose, Kc)
        dentro = z[0] > 0 and 0 <= uv[0, 0] < 1920 and 0 <= uv[0, 1] < 1080
        n_dr += int(dentro)
        env_px = Kc.fx * 0.9 / max(z[0], 1e-6)
        alturas = []
        for g in grupo:
            p = pos(P, g, t)
            uvp, zp = projetar(np.array([[p[0], p[1], 0.0], [p[0], p[1], 1.75]]), pose, Kc)
            alturas.append(abs(uvp[0, 1] - uvp[1, 1]))
        # camera do drone (gimbal) olhando para o centroide
        pg = Pose(dpos, olhar_para(dpos, [c[0], c[1], 0.9]))
        dist_g = float(np.linalg.norm(dpos - np.array([c[0], c[1], 0.9])))
        linhas.append(dict(t=round(float(t), 3), inclinacao=round(inclinacao_graus(C, alvo), 1), C=np.round(C, 2).tolist(),
                           drone=np.round(dpos, 2).tolist(), drone_na_imagem=bool(dentro), drone_px=round(float(env_px), 1),
                           drone_uv=np.round(uv[0], 0).tolist(), dist_cam_grupo=round(float(np.hypot(*(C[:2] - c))), 1),
                           altura_pessoa_px=round(float(np.mean(alturas)), 0), v_drone=round(vh, 2), incl_drone=round(incl, 1),
                           gimbal_dist=round(dist_g, 1), gimbal_incl=round(inclinacao_graus(dpos, [c[0], c[1], 0.9]), 1)))
        if k % (2 * fps) == 0:
            amostras.append((t, pose, c, dpos, pg, Kc))
    plano["conferencia"] = {
        "quadros": len(linhas), "quadros_com_drone": n_dr,
        "inclinacao_graus": [min(l["inclinacao"] for l in linhas), max(l["inclinacao"] for l in linhas)],
        "dist_cam_grupo_m": [min(l["dist_cam_grupo"] for l in linhas), max(l["dist_cam_grupo"] for l in linhas)],
        "altura_camera_m": [round(min(l["C"][2] for l in linhas), 2), round(max(l["C"][2] for l in linhas), 2)],
        "altura_pessoa_px": [min(l["altura_pessoa_px"] for l in linhas), max(l["altura_pessoa_px"] for l in linhas)],
        "drone_px_quando_visivel": [min((l["drone_px"] for l in linhas if l["drone_na_imagem"]), default=0),
                                    max((l["drone_px"] for l in linhas if l["drone_na_imagem"]), default=0)],
        "gimbal_dist_m": [min(l["gimbal_dist"] for l in linhas), max(l["gimbal_dist"] for l in linhas)],
    }
    out = Path(a.saida)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(plano, indent=1, ensure_ascii=False), encoding="utf-8")
    out.with_name(out.stem + "_quadros.json").write_text(json.dumps(linhas, indent=0), encoding="utf-8")
    print(json.dumps(plano["conferencia"], ensure_ascii=False))
    print("pessoas no render:", len(sel), sel)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, (len(amostras) + 1) // 2, figsize=(5 * ((len(amostras) + 1) // 2), 6))
    for ax, (t, pose, c, dpos, pg, Kc) in zip(axs.flat, amostras):
        for pid in sel:
            p = pos(P, pid, t)
            uv, z = projetar(np.array([[p[0], p[1], 0.0], [p[0], p[1], 1.75]]), pose, Kc)
            if (z > 0).all():
                ax.plot(uv[:, 0], uv[:, 1], "-", color="red" if pid in grupo else "k", lw=2)
        g = np.arange(-60, 61, 5.0)
        for v in g:  # grade de 5 m no chao
            for seg in ([[c[0] + v, c[1] - 60, 0], [c[0] + v, c[1] + 60, 0]], [[c[0] - 60, c[1] + v, 0], [c[0] + 60, c[1] + v, 0]]):
                uv, z = projetar(np.array(seg), pose, Kc)
                if (z > 0).all():
                    ax.plot(uv[:, 0], uv[:, 1], color="0.85", lw=0.5, zorder=0)
        uv, z = projetar(dpos[None], pose, Kc)
        if z[0] > 0:
            ax.plot(uv[0, 0], uv[0, 1], "bx", ms=12, mew=3)
        ax.set_xlim(0, 1920); ax.set_ylim(1080, 0); ax.set_aspect("equal")
        ax.set_title(f"t = {t:.1f} s; incl. {inclinacao_graus(pose.C, pose.C + pose.R_wc[:, 2]):.0f}°; hfov {Kc.hfov_graus:.0f}°", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=80)


if __name__ == "__main__":
    main()
