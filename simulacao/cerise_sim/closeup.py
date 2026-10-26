"""Geometria do close-up "cinegrafista" (camera baixa e obliqua perto de um grupo + um quadricoptero em voo baixo).

Sem dependencias do Isaac: usado por isaac/closeup_isaac.py (render) e por scripts/planejar_closeup.py (conferencia do
enquadramento por projecao pinhole antes do render). Coordenadas ENU locais do cenario (x leste, y norte, z cima, m).

Plano (JSON gerado por scripts/planejar_closeup.py):
  grupo:            ids das pessoas do grupo seguido pela camera
  centroide_pred:   [[t, x, y], ...] centroide previsto do grupo (das posicoes reais do voo run01)
  camera:           raio/altura/azimute inicial e final da orbita, alvo vertical, hfov
  drone:            pontos de passagem [[t, x, y, z], ...] (interpolacao suave), escala do modelo
  gimbal:           hfov e resolucao da camera do drone (aponta para o centroide do grupo)
"""
from __future__ import annotations

import math

import numpy as np


def suave(u):
    """smoothstep C1 em [0, 1]."""
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def interp_tabela(tab, t):
    a = np.asarray(tab, float)
    return np.array([np.interp(t, a[:, 0], a[:, k]) for k in range(1, a.shape[1])])


class PlanoCloseup:
    def __init__(self, plano: dict):
        self.p = plano
        self.cam = plano["camera"]
        self.dr = plano["drone"]
        self.t0, self.t1 = plano["t_inicio"], plano["t_fim"]

    # ------------------------------------------------------------------ drone
    def drone(self, t):
        """posicao (x, y, z), rumo (graus, a partir do leste) e inclinacao (graus, para a frente) do drone em t."""
        wp = np.asarray(self.dr["pontos"], float)
        # Catmull-Rom uniforme no parametro tempo (curva suave que passa pelos pontos)
        tt = wp[:, 0]
        t = min(max(t, tt[0]), tt[-1])
        i = int(np.clip(np.searchsorted(tt, t) - 1, 0, len(tt) - 2))
        p0, p1 = wp[max(i - 1, 0), 1:], wp[i, 1:]
        p2, p3 = wp[i + 1, 1:], wp[min(i + 2, len(wp) - 1), 1:]
        u = (t - tt[i]) / max(tt[i + 1] - tt[i], 1e-9)
        pos = 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)
        return pos

    def drone_estado(self, t, h=0.05):
        pos = self.drone(t)
        v = (self.drone(t + h) - self.drone(t - h)) / (2 * h)
        a = (self.drone(t + h) - 2 * pos + self.drone(t - h)) / (h * h)
        vh = math.hypot(v[0], v[1])
        rumo = math.degrees(math.atan2(v[1], v[0])) if vh > 0.2 else self.dr.get("rumo_parado", 0.0)
        # multirrotor inclina para a frente proporcionalmente ao arrasto (~v) e a aceleracao longitudinal
        a_long = (a[0] * v[0] + a[1] * v[1]) / max(vh, 1e-6)
        incl = float(np.clip(1.6 * vh + 3.0 * a_long, -15, 15))
        return pos, rumo, incl, vh

    # ------------------------------------------------------------------ camera do cinegrafista
    def peso_passagem(self, t):
        """0..1: quanto a camera se afasta e sobe o enquadramento para incluir o drone (0 = so o grupo)."""
        c = self.cam
        sobe = suave((t - c["tilt_ini"]) / max(c["tilt_dur"], 1e-6))
        desce = suave((t - c["tilt_volta"]) / max(c["tilt_dur"], 1e-6)) if c.get("tilt_volta") else 0.0
        return sobe - desce

    def hfov(self, t):
        """campo de visao horizontal (graus): abre durante a passagem do drone (zoom out)."""
        c = self.cam
        return c["hfov"] + (c.get("hfov_passagem", c["hfov"]) - c["hfov"]) * self.peso_passagem(t)

    def camera(self, t, centro, drone=None):
        """centro = centroide (suavizado) do grupo no chao; drone = posicao do drone (padrao: a do plano).
        Orbita lenta em torno do grupo; durante a passagem do drone a camera recua (raio_extra), desce
        (altura_extra) e levanta o enquadramento ate caberem os pes do grupo e o drone. Devolve (C, alvo)."""
        c = self.cam
        u = (t - self.t0) / max(self.t1 - self.t0, 1e-6)
        s = suave(u) * 0.35 + 0.65 * u  # orbita quase linear, com partida suave
        w = self.peso_passagem(t)
        phi = math.radians(c["azimute_ini"] + (c["azimute_fim"] - c["azimute_ini"]) * s)
        raio = c["raio_ini"] + (c["raio_fim"] - c["raio_ini"]) * s + c.get("raio_extra", 0.0) * w
        alt = c["altura_ini"] + (c["altura_fim"] - c["altura_ini"]) * s + c.get("altura_extra", 0.0) * w
        C = np.array([centro[0] + raio * math.cos(phi), centro[1] + raio * math.sin(phi), alt])
        G = np.array([centro[0], centro[1], c["alvo_z"]])
        if w <= 1e-6:
            return C, G
        D = self.drone(t) if drone is None else np.asarray(drone, float)

        def ang(P):
            v = np.asarray(P, float) - C
            return math.atan2(v[1], v[0]), math.atan2(v[2], math.hypot(v[0], v[1]))
        az_g, el_g = ang(G)
        az_d, el_d = ang(D)
        _, el_pes = ang([centro[0], centro[1], 0.0])
        az_m = az_g + 0.5 * math.atan2(math.sin(az_d - az_g), math.cos(az_d - az_g))
        fp = c.get("fracao_pes", 0.5)  # > 0,5 deixa mais margem abaixo dos pes do grupo
        el_m = fp * el_pes + (1 - fp) * el_d
        az = az_g + w * math.atan2(math.sin(az_m - az_g), math.cos(az_m - az_g))
        el = el_g + w * (el_m - el_g)
        alvo = C + 10.0 * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        return C, alvo


def inclinacao_graus(C, alvo):
    """angulo do eixo optico abaixo da horizontal (graus; > 0 = olhando para baixo)."""
    f = np.asarray(alvo, float) - np.asarray(C, float)
    return math.degrees(math.atan2(-f[2], math.hypot(f[0], f[1])))
