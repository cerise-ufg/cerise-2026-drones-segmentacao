"""Trajetorias temporizadas dos drones a partir das rotas boustrophedon do cerise_drones.

Modelo (o mesmo de cerise_drones.cobertura.simular_missao, sem trocas de bateria):
- decolagem vertical na base ate a altitude (vel_subida_ms);
- pernas em linha reta a velocidade de cruzeiro; em cada curva (> 20 graus) o drone para no vertice
  e gira o rumo durante `tempo_curva_s` (a camera acompanha o rumo: topo da imagem = direcao de voo);
- pouso vertical no ultimo ponto (vel_descida_ms).
A trajetoria e uma lista de quadros-chave (t, x, y, z, yaw) com interpolacao linear; `cobre` indica se
o intervalo que comeca no quadro-chave e de cobertura (camera "ativa" no plano original).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _rumo(a, b) -> float:
    d = np.asarray(b, float) - np.asarray(a, float)
    return math.degrees(math.atan2(d[1], d[0]))


def _desenrolar(yaw_ant: float, yaw: float) -> float:
    """Menor rotacao: devolve yaw equivalente a `yaw` mais proximo de yaw_ant."""
    d = (yaw - yaw_ant + 180.0) % 360.0 - 180.0
    return yaw_ant + d


@dataclass
class Trajetoria:
    drone_id: str
    t: np.ndarray            # (M,)
    xyz: np.ndarray          # (M,3)
    yaw: np.ndarray          # (M,) graus, desenrolado (continuo)
    cobre: np.ndarray        # (M,) bool: intervalo [t_k, t_k+1) e de cobertura
    fase: list = field(default_factory=list)  # (M,) rotulo do intervalo

    @property
    def duracao_s(self) -> float:
        return float(self.t[-1])

    def amostrar(self, tt):
        tt = np.clip(np.atleast_1d(np.asarray(tt, float)), self.t[0], self.t[-1])
        x = np.interp(tt, self.t, self.xyz[:, 0])
        y = np.interp(tt, self.t, self.xyz[:, 1])
        z = np.interp(tt, self.t, self.xyz[:, 2])
        yaw = np.interp(tt, self.t, self.yaw)
        k = np.clip(np.searchsorted(self.t, tt, side="right") - 1, 0, len(self.t) - 2)
        cobre = self.cobre[k]
        fase = [self.fase[i] for i in k]
        return x, y, z, yaw, cobre, fase

    def para_dict(self) -> dict:
        return {"drone_id": self.drone_id, "t": self.t.round(4).tolist(), "xyz": self.xyz.round(4).tolist(),
                "yaw": self.yaw.round(4).tolist(), "cobre": self.cobre.astype(int).tolist(), "fase": self.fase}

    @classmethod
    def de_dict(cls, d: dict) -> "Trajetoria":
        return cls(d["drone_id"], np.array(d["t"], float), np.array(d["xyz"], float), np.array(d["yaw"], float),
                   np.array(d["cobre"], bool), list(d["fase"]))


def construir(drone_id: str, pontos_xy, cobre, altitude_m: float, velocidade_ms: float,
              vel_subida_ms: float = 3.0, vel_descida_ms: float = 2.0, tempo_curva_s: float = 4.0,
              limiar_curva_graus: float = 20.0, t0: float = 0.0, decolar: bool = True) -> Trajetoria:
    P = np.asarray(pontos_xy, float)
    C = np.asarray(cobre, bool)
    # remove pernas nulas
    keep = [0]
    cob = []
    for k in range(1, len(P)):
        if np.linalg.norm(P[k] - P[keep[-1]]) > 1e-3:
            keep.append(k)
            cob.append(bool(C[k - 1]))
    P = P[keep]
    T, X, Y, Z, W, CB, F = [], [], [], [], [], [], []

    def add(t, p, z, w, cb, f):
        T.append(t); X.append(p[0]); Y.append(p[1]); Z.append(z); W.append(w); CB.append(cb); F.append(f)

    t = t0
    yaw = _rumo(P[0], P[1]) if len(P) > 1 else 0.0
    if decolar:
        add(t, P[0], 0.0, yaw, False, "decolagem")
        t += altitude_m / vel_subida_ms
    for k in range(len(P) - 1):
        a, b = P[k], P[k + 1]
        novo = _desenrolar(yaw, _rumo(a, b))
        if k > 0 and abs(novo - yaw) > limiar_curva_graus:
            add(t, a, altitude_m, yaw, False, "curva")
            t += tempo_curva_s
        yaw = novo
        add(t, a, altitude_m, yaw, cob[k], "cobertura" if cob[k] else "transito")
        t += float(np.linalg.norm(b - a)) / velocidade_ms
    add(t, P[-1], altitude_m, yaw, False, "pouso")
    t += altitude_m / vel_descida_ms
    add(t, P[-1], 0.0, yaw, False, "fim")
    return Trajetoria(drone_id, np.array(T), np.column_stack([X, Y, Z]), np.array(W), np.array(CB, bool), F)
