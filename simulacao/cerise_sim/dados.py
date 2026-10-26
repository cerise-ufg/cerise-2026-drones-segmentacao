"""Formato comum do dataset gravado pelos simuladores (Isaac Sim ou Gazebo) e leitores.

<saida>/
  meta.json                       simulador, cenario, fps, origem ENU, cameras {id: {intrinsecos, tipo}}
  gabarito_pessoas.csv            frame,t,pessoa_id,x,y,z,yaw     (posicoes REAIS gravadas no simulador)
  <cam>/rgb/<frame:06d>.jpg       (ou .png)
  <cam>/poses.csv                 frame,t,cx,cy,cz,qw,qx,qy,qz,yaw,cobre   (R_wc OpenCV -> ENU, ver camera.py)
  <cam>/caixas_gt.csv             frame,pessoa_id,x1,y1,x2,y2,oclusao      (Replicator bounding_box_2d_tight)
<cam> = id do drone (D1..D4) ou "visao_geral".
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camera import Intrinsecos, Pose, matriz_de_quat

CAB_POSES = ["frame", "t", "cx", "cy", "cz", "qw", "qx", "qy", "qz", "yaw", "cobre"]
CAB_CAIXAS = ["frame", "pessoa_id", "x1", "y1", "x2", "y2", "oclusao"]
CAB_GAB = ["frame", "t", "pessoa_id", "x", "y", "z", "yaw"]


def escrever_csv(caminho, cabecalho, linhas):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cabecalho)
        w.writerows(linhas)


def ler_csv(caminho) -> list[dict]:
    with open(caminho, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass
class Quadro:
    frame: int
    t: float
    pose: Pose
    yaw: float
    cobre: bool
    imagem: Path | None


class Execucao:
    def __init__(self, pasta):
        self.pasta = Path(pasta)
        self.meta = json.loads((self.pasta / "meta.json").read_text(encoding="utf-8"))
        self._cache_gab = None

    @property
    def fps(self) -> float:
        return float(self.meta["fps"])

    def cameras(self, tipo: str | None = "drone") -> list[str]:
        return [c for c, d in self.meta["cameras"].items() if tipo is None or d.get("tipo") == tipo]

    def intrinsecos(self, cam: str) -> Intrinsecos:
        return Intrinsecos.de_dict(self.meta["cameras"][cam]["intrinsecos"])

    def quadros(self, cam: str) -> list[Quadro]:
        out = []
        pasta_rgb = self.pasta / cam / "rgb"
        for r in ler_csv(self.pasta / cam / "poses.csv"):
            f = int(r["frame"])
            q = [float(r[k]) for k in ("qw", "qx", "qy", "qz")]
            pose = Pose(np.array([float(r["cx"]), float(r["cy"]), float(r["cz"])]), matriz_de_quat(q))
            img = None
            for ext in (".jpg", ".png"):
                p = pasta_rgb / f"{f:06d}{ext}"
                if p.exists():
                    img = p
                    break
            out.append(Quadro(f, float(r["t"]), pose, float(r.get("yaw", 0) or 0), bool(int(float(r.get("cobre", 1) or 0))), img))
        return out

    def caixas_gt(self, cam: str) -> dict:
        """frame -> (ids list, caixas (n,4), oclusao (n,))"""
        arq = self.pasta / cam / "caixas_gt.csv"
        out: dict = {}
        if not arq.exists():
            return out
        for r in ler_csv(arq):
            f = int(r["frame"])
            ids, bx, oc = out.setdefault(f, ([], [], []))
            ids.append(r["pessoa_id"])
            bx.append([float(r[k]) for k in ("x1", "y1", "x2", "y2")])
            oc.append(float(r.get("oclusao", 0) or 0))
        return {f: (ids, np.array(bx, float).reshape(-1, 4), np.array(oc, float)) for f, (ids, bx, oc) in out.items()}

    def gabarito(self) -> dict:
        """frame -> {pessoa_id: (x, y)}"""
        if self._cache_gab is None:
            g: dict = {}
            for r in ler_csv(self.pasta / "gabarito_pessoas.csv"):
                g.setdefault(int(r["frame"]), {})[r["pessoa_id"]] = (float(r["x"]), float(r["y"]))
            self._cache_gab = g
        return self._cache_gab

    def pessoas(self) -> list[str]:
        if "pessoas" in self.meta:
            return list(self.meta["pessoas"])
        ids = set()
        for d in self.gabarito().values():
            ids |= set(d)
        return sorted(ids)
