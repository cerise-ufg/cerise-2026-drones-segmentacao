"""Plano B: captura das cameras dos drones no Gazebo Harmonic via gz-transport (Python), no formato comum
de cerise_sim/dados.py. Roda dentro do conteiner cerise26_gzpx4 (ver rodar_sim_gz.sh).

- imagem RGB de /cerise/<id>/camera (gz.msgs.Image, carimbo = tempo de simulacao);
- pose do x500 de /world/ufg/dynamic_pose/info (interpolada no carimbo da imagem) composta com a montagem
  da camera (px4_instancias.json: camera_no_modelo) -> pose completa (inclui rolagem/arfagem do voo);
- caixas 2D de /cerise/<id>/caixas (sensor boundingbox_camera; rotulo do sistema Label -> pessoa_id);
- gabarito das pessoas: trajetorias roteirizadas dos atores = plano (exato no tempo de simulacao).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, "/sim")
from cerise_sim.camera import Intrinsecos, matriz_de_quat, olhar_para, quat_de_matriz  # noqa: E402
from cerise_sim.dados import CAB_CAIXAS, CAB_GAB, CAB_POSES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mundo", required=True)
ap.add_argument("--cenario", required=True)
ap.add_argument("--saida", required=True)
ap.add_argument("--fps", type=float, default=5.0)
ap.add_argument("--duracao", type=float, default=0.0)
a = ap.parse_args()

from gz.msgs10.annotated_axis_aligned_2d_box_v_pb2 import AnnotatedAxisAligned2DBox_V  # noqa: E402
from gz.msgs10.image_pb2 import Image  # noqa: E402
from gz.msgs10.pose_v_pb2 import Pose_V  # noqa: E402
from gz.transport13 import Node  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

inst = json.loads((Path(a.mundo) / "px4_instancias.json").read_text())
cen = json.loads((Path(a.cenario) / "cenario.json").read_text())
pessoas = {p["id"]: np.asarray(p["kf"], float) for p in json.loads((Path(a.cenario) / "pessoas_plano.json").read_text())["pessoas"]
           if p["id"] in set(inst["pessoas"])}
rotulos = {int(k): v for k, v in inst["rotulos"].items()}
R_mc = np.array(inst["camera_no_modelo"]["R_modelo_cv"], float).T  # colunas = eixos da camera no modelo
t_mc = np.array(inst["camera_no_modelo"]["t"], float)
dur = a.duracao or inst["duracao_s"]
OUT = Path(a.saida)
drones = {i["drone"]: i for i in inst["instancias"]}
cam_d = {d["id"]: d for d in cen["drones"] if d["id"] in drones}

POSES = collections.defaultdict(lambda: collections.deque(maxlen=2000))  # modelo -> (t, p, q[w,x,y,z])
trava = threading.Lock()
arq = {}
for did in drones:
    (OUT / did / "rgb").mkdir(parents=True, exist_ok=True)
    arq[(did, "p")] = open(OUT / did / "poses.csv", "w"); arq[(did, "p")].write(",".join(CAB_POSES) + "\n")
    arq[(did, "c")] = open(OUT / did / "caixas_gt.csv", "w"); arq[(did, "c")].write(",".join(CAB_CAIXAS) + "\n")
gab = open(OUT / "gabarito_pessoas.csv", "w"); gab.write(",".join(CAB_GAB) + "\n")
gab_feitos = set()
pool = ThreadPoolExecutor(max_workers=3)
estat = collections.Counter()
t_max = [0.0]


def carimbo(h):
    return h.stamp.sec + h.stamp.nsec * 1e-9


def ao_pose(msg: Pose_V):
    t = carimbo(msg.header)
    t_max[0] = max(t_max[0], t)
    with trava:
        for p in msg.pose:
            if p.name.startswith("x500_cerise_"):
                POSES[p.name].append((t, np.array([p.position.x, p.position.y, p.position.z]),
                                      np.array([p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z])))


def pose_em(modelo, t):
    with trava:
        h = list(POSES.get(modelo, []))
    if not h:
        return None
    ts = np.array([x[0] for x in h])
    k = int(np.searchsorted(ts, t))
    if k <= 0:
        return h[0][1], h[0][2]
    if k >= len(h):
        return h[-1][1], h[-1][2]
    (t0, p0, q0), (t1, p1, q1) = h[k - 1], h[k]
    w = (t - t0) / max(t1 - t0, 1e-9)
    q1 = q1 if np.dot(q0, q1) >= 0 else -q1
    q = (1 - w) * q0 + w * q1
    return (1 - w) * p0 + w * p1, q / np.linalg.norm(q)


def quadro_de(t):
    return int(round(t * a.fps))


def gravar_gabarito(f, t):
    if f in gab_feitos:
        return
    gab_feitos.add(f)
    for pid, kf in pessoas.items():
        tt = min(max(t, kf[0, 0]), kf[-1, 0])
        x, y, w = (np.interp(tt, kf[:, 0], kf[:, c]) for c in (1, 2, 3))
        gab.write(f"{f},{t:.4f},{pid},{x:.3f},{y:.3f},0.000,{w:.2f}\n")


def fabrica_imagem(did):
    modelo = drones[did]["modelo_gz"]

    def cb(msg: Image):
        t = carimbo(msg.header)
        if t > dur:
            return
        f = quadro_de(t)
        pq = pose_em(modelo, t)
        if pq is None:
            estat["sem_pose"] += 1
            return
        p, q = pq
        R_wm = matriz_de_quat(q)
        R_wc = R_wm @ R_mc
        C = p + R_wm @ t_mc
        qc = quat_de_matriz(R_wc)
        yaw = math.degrees(math.atan2(R_wm[1, 0], R_wm[0, 0]))
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)[..., :3].copy()
        pool.submit(lambda a_=arr, f_=f: PILImage.fromarray(a_).save(OUT / did / "rgb" / f"{f_:06d}.jpg", quality=93))
        with trava:
            arq[(did, "p")].write(f"{f},{t:.4f},{C[0]:.4f},{C[1]:.4f},{C[2]:.4f},{qc[0]:.6f},{qc[1]:.6f},{qc[2]:.6f},{qc[3]:.6f},"
                                  f"{yaw:.3f},{int(C[2] > 5.0)}\n")
            gravar_gabarito(f, t)
        estat[f"img_{did}"] += 1
    return cb


def fabrica_caixas(did):
    def cb(msg: AnnotatedAxisAligned2DBox_V):
        t = carimbo(msg.header)
        if t > dur:
            return
        f = quadro_de(t)
        with trava:
            for b in msg.annotated_box:
                pid = rotulos.get(int(b.label))
                if pid is None:
                    continue
                arq[(did, "c")].write(f"{f},{pid},{b.box.min_corner.x:.1f},{b.box.min_corner.y:.1f},"
                                      f"{b.box.max_corner.x:.1f},{b.box.max_corner.y:.1f},0.000\n")
        estat[f"cx_{did}"] += 1
    return cb


def ao_visao_geral(msg: Image):
    t = carimbo(msg.header)
    if t > dur:
        return
    f = quadro_de(t)
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)[..., :3].copy()
    pool.submit(lambda a_=arr, f_=f: PILImage.fromarray(a_).save(OUT / "visao_geral" / "rgb" / f"{f_:06d}.jpg", quality=90))
    with trava:
        C = cen["visao_geral"]["posicao"]
        q = quat_de_matriz(olhar_para(C, cen["visao_geral"]["alvo"]))
        arq[("visao_geral", "p")].write(f"{f},{t:.4f},{C[0]:.4f},{C[1]:.4f},{C[2]:.4f},{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f},0,0\n")


(OUT / "visao_geral" / "rgb").mkdir(parents=True, exist_ok=True)
arq[("visao_geral", "p")] = open(OUT / "visao_geral" / "poses.csv", "w"); arq[("visao_geral", "p")].write(",".join(CAB_POSES) + "\n")
node = Node()
node.subscribe(Image, "/cerise/visao_geral/camera", ao_visao_geral)
node.subscribe(Pose_V, "/world/ufg/dynamic_pose/info", ao_pose)
for did in drones:
    node.subscribe(Image, f"/cerise/{did}/camera", fabrica_imagem(did))
    node.subscribe(AnnotatedAxisAligned2DBox_V, f"/cerise/{did}/caixas", fabrica_caixas(did))
meta = {"simulador": "gazebo-harmonic + px4-sitl", "cenario": cen["nome"], "fps": a.fps, "origem_enu": cen["origem_enu"],
        "cameras": {**{did: {"tipo": "drone", "intrinsecos": cam_d[did]["camera"]["intrinsecos"]} for did in drones},
                    "visao_geral": {"tipo": "visao_geral", "intrinsecos": cen["visao_geral"]["intrinsecos"]}},
        "pessoas": list(pessoas), "modo_pessoas": "atores_gazebo_trajetoria_roteirizada"}
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
print("capturando ate t_sim =", dur, flush=True)
t0 = time.time()
while t_max[0] < dur:
    time.sleep(2.0)
    print(f"t_sim {t_max[0]:.1f} s; {dict(estat)}; {time.time() - t0:.0f} s de relogio", flush=True)
pool.shutdown(wait=True)
with trava:
    for f in list(arq.values()) + [gab]:
        f.close()
print("fim", dict(estat), flush=True)
