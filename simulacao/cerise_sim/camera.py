"""Modelo de camera pinhole, pose em nadir (ou obliqua) e projecao mundo <-> pixel.

Convencoes
- Mundo: ENU (x = leste, y = norte, z = cima), chao em z = 0.
- Camera (OpenCV): x = direita da imagem, y = para baixo na imagem, z = eixo optico (para frente).
- R_wc: rotacao camera -> mundo (colunas = eixos da camera escritos no mundo); C = centro optico.
  p_cam = R_wc^T (P_mundo - C);  u = fx * x/z + cx;  v = fy * y/z + cy.
- yaw (graus): direcao de voo medida a partir do LESTE no sentido anti-horario (convencao ENU).
  rumo (graus, a partir do norte no sentido horario) = 90 - yaw.
- Em nadir, o topo da imagem aponta para a direcao de voo (mesma convencao de
  cerise_drones.agregacao.pixel_para_solo); `inclinacao_graus` > 0 inclina o eixo optico para a frente.
- USD/Isaac: a camera olha para -Z local com +Y para cima => R_usd = R_wc @ diag(1, -1, -1).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

ABERTURA_H_MM = 20.955  # abertura horizontal padrao das cameras USD (Isaac Sim)


@dataclass
class Intrinsecos:
    largura: int
    altura: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def de_hfov(cls, largura: int, altura: int, hfov_graus: float) -> "Intrinsecos":
        """Pixels quadrados, ponto principal no centro."""
        fx = (largura / 2.0) / math.tan(math.radians(hfov_graus) / 2.0)
        return cls(int(largura), int(altura), fx, fx, largura / 2.0, altura / 2.0)

    @property
    def hfov_graus(self) -> float:
        return math.degrees(2 * math.atan(self.largura / (2 * self.fx)))

    @property
    def vfov_graus(self) -> float:
        return math.degrees(2 * math.atan(self.altura / (2 * self.fy)))

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def usd(self, abertura_h_mm: float = ABERTURA_H_MM) -> dict:
        """Parametros da camera USD (mm) que reproduzem estes intrinsecos (pixels quadrados)."""
        foco_mm = self.fx * abertura_h_mm / self.largura
        return {"focalLength": foco_mm, "horizontalAperture": abertura_h_mm,
                "verticalAperture": abertura_h_mm * self.altura / self.largura}

    def para_dict(self) -> dict:
        d = asdict(self)
        d.update(hfov_graus=self.hfov_graus, vfov_graus=self.vfov_graus, usd=self.usd())
        return d

    @classmethod
    def de_dict(cls, d: dict) -> "Intrinsecos":
        return cls(int(d["largura"]), int(d["altura"]), float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]))


def rotacao_camera(yaw_graus: float, inclinacao_graus: float = 0.0) -> np.ndarray:
    """R_wc (OpenCV -> mundo ENU) para camera que olha para baixo, com o topo da imagem na direcao
    yaw; inclinacao > 0 levanta o eixo optico para a frente (0 = nadir)."""
    psi = math.radians(yaw_graus)
    th = math.radians(inclinacao_graus)
    f = np.array([math.cos(psi), math.sin(psi), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    z = -math.cos(th) * up + math.sin(th) * f          # eixo optico
    y = -(math.cos(th) * f + math.sin(th) * up)        # para baixo na imagem
    x = np.cross(y, z)                                 # direita da imagem
    return np.column_stack([x, y, z])


def r_usd_de_cv(R_wc: np.ndarray) -> np.ndarray:
    return R_wc @ np.diag([1.0, -1.0, -1.0])


def r_cv_de_usd(R_usd: np.ndarray) -> np.ndarray:
    return R_usd @ np.diag([1.0, -1.0, -1.0])


def quat_de_matriz(R: np.ndarray) -> np.ndarray:
    """(w, x, y, z), norma 1, w >= 0."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def matriz_de_quat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def olhar_para(origem, alvo, cima=(0.0, 0.0, 1.0)) -> np.ndarray:
    """R_wc (OpenCV) de uma camera em `origem` olhando para `alvo` (visao geral em perspectiva)."""
    o = np.asarray(origem, float); a = np.asarray(alvo, float)
    z = a - o; z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(cima, float)); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


@dataclass
class Pose:
    C: np.ndarray        # (3,) centro optico no mundo
    R_wc: np.ndarray     # (3,3)

    def mundo_para_camera(self, P) -> np.ndarray:
        P = np.atleast_2d(np.asarray(P, float))
        return (P - self.C) @ self.R_wc  # == (R_wc^T (P - C))^T


def projetar(P, pose: Pose, K: Intrinsecos):
    """Pontos do mundo (n,3) -> pixels (n,2) e profundidade z_cam (n,)."""
    pc = pose.mundo_para_camera(P)
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K.fx * pc[:, 0] / z + K.cx
        v = K.fy * pc[:, 1] / z + K.cy
    return np.column_stack([u, v]), z


def pixel_para_chao(uv, pose: Pose, K: Intrinsecos, z_plano: float = 0.0) -> np.ndarray:
    """Interseccao do raio de cada pixel com o plano horizontal z = z_plano. (n,2) -> (n,3);
    raios que nao descem ate o plano viram NaN."""
    uv = np.atleast_2d(np.asarray(uv, float))
    d_cam = np.column_stack([(uv[:, 0] - K.cx) / K.fx, (uv[:, 1] - K.cy) / K.fy, np.ones(len(uv))])
    d_w = d_cam @ pose.R_wc.T
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (z_plano - pose.C[2]) / d_w[:, 2]
    P = pose.C + lam[:, None] * d_w
    P[~(lam > 0)] = np.nan
    return P


def pegada_no_chao(pose: Pose, K: Intrinsecos, z_plano: float = 0.0) -> np.ndarray:
    """Quatro cantos da imagem projetados no chao (4,3)."""
    cantos = np.array([[0, 0], [K.largura, 0], [K.largura, K.altura], [0, K.altura]], float)
    return pixel_para_chao(cantos, pose, K, z_plano)


def pose_drone(x: float, y: float, z: float, yaw_graus: float, inclinacao_graus: float = 0.0) -> Pose:
    return Pose(np.array([x, y, z], float), rotacao_camera(yaw_graus, inclinacao_graus))
