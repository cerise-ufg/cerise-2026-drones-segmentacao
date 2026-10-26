"""Rastreamento multi-alvo.

RastreadorChao  : no plano do chao (ENU), usado quando ha pose da camera (simulacao ou voo com
                  telemetria). Associacao Hungara por distancia com portao que cresce com o tempo
                  desde a ultima atualizacao (velocidade maxima de caminhada); posicao suavizada
                  e velocidade estimada (alfa-beta). Faixas so contam depois de `min_hits` deteccoes.
RastreadorImagem: no plano da imagem (videos reais sem pose): custo 1 - IoU combinado com distancia
                  normalizada dos centros (alvos pequenos), estilo SORT sem filtro de Kalman.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Faixa:
    id: int
    t: list = field(default_factory=list)
    xy: list = field(default_factory=list)      # posicoes medidas (deteccoes associadas)
    frames: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    caixas: list = field(default_factory=list)  # caixas (pixels) das deteccoes
    pos: np.ndarray | None = None               # estimativa suavizada
    vel: np.ndarray | None = None
    t_ult: float = 0.0
    hits: int = 0

    @property
    def t_ini(self) -> float:
        return self.t[0]

    @property
    def t_fim(self) -> float:
        return self.t[-1]

    def posicao_representativa(self) -> np.ndarray:
        return np.median(np.asarray(self.xy, float), axis=0)

    def posicao_em(self, tt):
        a = np.asarray(self.xy, float)
        t = np.asarray(self.t, float)
        return np.column_stack([np.interp(tt, t, a[:, 0]), np.interp(tt, t, a[:, 1])])


class RastreadorChao:
    def __init__(self, dist_max_m: float = 2.0, vel_max_ms: float = 2.5, min_hits: int = 3,
                 max_idade_s: float = 2.0, alfa: float = 0.6, beta: float = 0.2):
        self.dist_max = dist_max_m
        self.vel_max = vel_max_ms
        self.min_hits = min_hits
        self.max_idade = max_idade_s
        self.alfa, self.beta = alfa, beta
        self.ativas: list[Faixa] = []
        self.encerradas: list[Faixa] = []
        self._prox = 0

    def _prever(self, f: Faixa, t: float) -> np.ndarray:
        dt = t - f.t_ult
        v = f.vel if f.vel is not None else np.zeros(2)
        return f.pos + v * min(dt, 1.0)

    def atualizar(self, t: float, xy, scores=None, frame: int = -1, caixas=None) -> list[int]:
        xy = np.asarray(xy, float).reshape(-1, 2)
        n = len(xy)
        scores = np.ones(n) if scores is None else np.asarray(scores, float)
        caixas = np.zeros((n, 4)) if caixas is None else np.asarray(caixas, float).reshape(-1, 4)
        # encerra faixas velhas
        vivas = []
        for f in self.ativas:
            (vivas if t - f.t_ult <= self.max_idade else self.encerradas).append(f)
        self.ativas = vivas
        ids = [-1] * n
        if self.ativas and n:
            P = np.array([self._prever(f, t) for f in self.ativas])
            D = np.linalg.norm(P[:, None, :] - xy[None, :, :], axis=2)
            portao = np.array([self.dist_max + self.vel_max * max(0.0, t - f.t_ult) * 0.5 for f in self.ativas])
            C = np.where(D <= portao[:, None], D, 1e6)
            li, co = linear_sum_assignment(C)
            for i, j in zip(li, co):
                if C[i, j] >= 1e6:
                    continue
                f = self.ativas[i]
                dt = max(t - f.t_ult, 1e-3)
                pred = P[i]
                r = xy[j] - pred
                f.pos = pred + self.alfa * r
                f.vel = (f.vel if f.vel is not None else np.zeros(2)) + self.beta * r / dt
                sp = np.linalg.norm(f.vel)
                if sp > self.vel_max:
                    f.vel *= self.vel_max / sp
                f.t_ult = t
                f.hits += 1
                f.t.append(t); f.xy.append(xy[j].tolist()); f.frames.append(frame)
                f.scores.append(float(scores[j])); f.caixas.append(caixas[j].tolist())
                ids[j] = f.id
        for j in range(n):
            if ids[j] == -1:
                f = Faixa(self._prox, [t], [xy[j].tolist()], [frame], [float(scores[j])], [caixas[j].tolist()],
                          xy[j].copy(), np.zeros(2), t, 1)
                self._prox += 1
                self.ativas.append(f)
                ids[j] = f.id
        return ids

    def finalizar(self) -> list[Faixa]:
        self.encerradas += self.ativas
        self.ativas = []
        return self.confirmadas()

    def confirmadas(self) -> list[Faixa]:
        return [f for f in self.encerradas + self.ativas if f.hits >= self.min_hits]


def iou_matriz(a, b) -> np.ndarray:
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


class RastreadorImagem:
    """Rastreio no plano da imagem (sem pose). Cada faixa guarda centros em pixels em `xy`."""

    def __init__(self, iou_min: float = 0.1, dist_norm_max: float = 1.5, min_hits: int = 3, max_idade_quadros: int = 10):
        self.iou_min, self.dist_norm_max = iou_min, dist_norm_max
        self.min_hits, self.max_idade = min_hits, max_idade_quadros
        self.ativas: list[Faixa] = []
        self.encerradas: list[Faixa] = []
        self._prox = 0
        self._q = 0

    def atualizar(self, caixas, scores=None, frame: int | None = None, t: float | None = None) -> list[int]:
        self._q = self._q + 1 if frame is None else frame
        t = float(self._q) if t is None else t
        caixas = np.asarray(caixas, float).reshape(-1, 4)
        n = len(caixas)
        scores = np.ones(n) if scores is None else np.asarray(scores, float)
        vivas = []
        for f in self.ativas:
            (vivas if self._q - f.frames[-1] <= self.max_idade else self.encerradas).append(f)
        self.ativas = vivas
        ids = [-1] * n
        if self.ativas and n:
            B = np.array([f.caixas[-1] for f in self.ativas])
            iou = iou_matriz(B, caixas)
            cb = np.column_stack([(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2])
            cd = np.column_stack([(caixas[:, 0] + caixas[:, 2]) / 2, (caixas[:, 1] + caixas[:, 3]) / 2])
            esc = np.maximum(np.hypot(B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]), 4.0)
            dn = np.linalg.norm(cb[:, None] - cd[None], axis=2) / esc[:, None]
            ok = (iou >= self.iou_min) | (dn <= self.dist_norm_max)
            C = np.where(ok, (1 - iou) + 0.5 * np.minimum(dn, 3), 1e6)
            li, co = linear_sum_assignment(C)
            for i, j in zip(li, co):
                if C[i, j] >= 1e6:
                    continue
                f = self.ativas[i]
                f.hits += 1
                f.t.append(t); f.frames.append(self._q); f.caixas.append(caixas[j].tolist())
                f.xy.append(cd[j].tolist()); f.scores.append(float(scores[j]))
                ids[j] = f.id
        for j in range(n):
            if ids[j] == -1:
                c = [(caixas[j, 0] + caixas[j, 2]) / 2, (caixas[j, 1] + caixas[j, 3]) / 2]
                f = Faixa(self._prox, [t], [c], [self._q], [float(scores[j])], [caixas[j].tolist()], np.array(c), None, t, 1)
                self._prox += 1
                self.ativas.append(f)
                ids[j] = f.id
        return ids

    def finalizar(self) -> list[Faixa]:
        self.encerradas += self.ativas
        self.ativas = []
        return [f for f in self.encerradas if f.hits >= self.min_hits]
