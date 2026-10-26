"""Plano B: gera o mundo do Gazebo Harmonic (gz-sim 8) a partir do mesmo cenario do Isaac Sim.

Saida (pasta --saida):
  mundo_ufg.sdf                 mundo com coordenadas esfericas na origem ENU do cenario, sol, terreno
                                (malhas OBJ com texturas procedurais), edificios (box), arvores, atores
                                caminhando (trajetorias temporizadas = gabarito exato) e os N drones x500
                                com camera em nadir (sensor camera + boundingbox_camera)
  modelos/terreno/*.obj         malhas do terreno (uma por material) com UV
  modelos/x500_cerise_<id>/     x500 do PX4 (incluido por URI) + camera nadir com os intrinsecos do drone
  px4_instancias.json           instancia PX4, nome do modelo, porta MAVLink e .plan de cada drone
Uso: python gazebo/gerar_mundo.py --cenario cenario/reduzido --saida gazebo/mundo_reduzido [--fps 5]
Depende so de numpy (sem shapely); roda no Windows, no Spark ou dentro do conteiner do Gazebo.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from cerise_sim.camera import Intrinsecos  # noqa: E402

ATOR_ANDANDO = "https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae"
ATOR_PARADO = "https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/stand.dae"
ALTURA_QUADRIL_ATOR = 1.0  # a origem do walk.dae fica no quadril (exemplo oficial actor.sdf usa z = 1.0)


def obj(caminho: Path, V, F, UV):
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("# cerise_sim terreno\n")
        for v in V:
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for t in UV:
            f.write(f"vt {t[0]:.4f} {t[1]:.4f}\n")
        f.write("vn 0 0 1\n")
        for a, b, c in F:
            f.write(f"f {a + 1}/{a + 1}/1 {b + 1}/{b + 1}/1 {c + 1}/{c + 1}/1\n")


def material_pbr(textura: str | None, cor=(0.5, 0.5, 0.5), rug=0.8) -> str:
    alb = f"<albedo_map>{textura}</albedo_map>" if textura else ""
    c = " ".join(f"{v:.3f}" for v in cor)
    return (f"<material><ambient>{c} 1</ambient><diffuse>{c} 1</diffuse><specular>0.05 0.05 0.05 1</specular>"
            f"<pbr><metal>{alb}<roughness>{rug}</roughness><metalness>0</metalness></metal></pbr></material>")


def modelo_drone(pasta: Path, did: str, K: Intrinsecos, fps: float, incl_graus: float):
    """x500 do PX4 incluido por URI (model://x500 vem de PX4-Autopilot/Tools/simulation/gz/models) +
    camera nadir. Pitch +90 graus no SDF = eixo X da camera (direcao de visada) apontando para -Z."""
    m = pasta / f"x500_cerise_{did}"
    m.mkdir(parents=True, exist_ok=True)
    (m / "model.config").write_text(
        f"<?xml version='1.0'?><model><name>x500_cerise_{did}</name><version>1.0</version><sdf version='1.9'>model.sdf</sdf>"
        f"<description>x500 + camera nadir ({K.largura}x{K.altura}, HFOV {K.hfov_graus:.1f})</description></model>",
        encoding="utf-8")
    pitch = math.radians(90.0 - incl_graus)
    cam = f"""
      <sensor name="camera_nadir" type="camera">
        <pose>0 0 -0.08 0 {pitch:.6f} 0</pose>
        <camera>
          <horizontal_fov>{math.radians(K.hfov_graus):.6f}</horizontal_fov>
          <image><width>{K.largura}</width><height>{K.altura}</height><format>R8G8B8</format></image>
          <clip><near>0.3</near><far>3000</far></clip>
        </camera>
        <always_on>1</always_on><update_rate>{fps}</update_rate><visualize>false</visualize>
        <topic>/cerise/{did}/camera</topic>
      </sensor>
      <sensor name="caixas_nadir" type="boundingbox_camera">
        <pose>0 0 -0.08 0 {pitch:.6f} 0</pose>
        <topic>/cerise/{did}/caixas</topic>
        <camera>
          <box_type>2d</box_type>
          <horizontal_fov>{math.radians(K.hfov_graus):.6f}</horizontal_fov>
          <image><width>{K.largura}</width><height>{K.altura}</height></image>
          <clip><near>0.3</near><far>3000</far></clip>
        </camera>
        <always_on>1</always_on><update_rate>{fps}</update_rate>
      </sensor>"""
    (m / "model.sdf").write_text(f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="x500_cerise_{did}">
    <include merge="true"><uri>model://x500</uri></include>
    <link name="camera_link">
      <pose>0 0 -0.05 0 0 0</pose>
      <inertial><mass>0.02</mass><inertia><ixx>1e-5</ixx><iyy>1e-5</iyy><izz>1e-5</izz></inertia></inertial>
      {cam}
    </link>
    <joint name="camera_joint" type="fixed"><parent>base_link</parent><child>camera_link</child></joint>
  </model>
</sdf>
""", encoding="utf-8")


def ator(p: dict, duracao: float, rotulo: int) -> str:
    """Ator com trajetoria temporizada; `rotulo` (1..255) unico por pessoa identifica a pessoa nas caixas
    do sensor boundingbox_camera (sistema Label)."""
    kf = np.asarray(p["kf"], float)
    parado = p["tipo"] in ("parado", "grupo_parado")
    pontos = []
    for t, x, y, w in kf:
        if t > duracao + 1:
            break
        pontos.append(f"<waypoint><time>{t:.2f}</time><pose>{x:.3f} {y:.3f} {ALTURA_QUADRIL_ATOR} 0 0 {math.radians(w):.4f}</pose></waypoint>")
    if len(pontos) < 2:
        x, y, w = kf[0, 1], kf[0, 2], kf[0, 3]
        pontos = [f"<waypoint><time>{t:.2f}</time><pose>{x:.3f} {y:.3f} {ALTURA_QUADRIL_ATOR} 0 0 {math.radians(w):.4f}</pose></waypoint>"
                  for t in (0.0, duracao)]
    skin = ATOR_PARADO if parado else ATOR_ANDANDO
    nome_anim = "parado" if parado else "andando"
    return f"""
    <actor name="{p['id']}">
      <skin><filename>{skin}</filename><scale>1.0</scale></skin>
      <animation name="{nome_anim}"><filename>{skin}</filename><interpolate_x>{'false' if parado else 'true'}</interpolate_x></animation>
      <script><loop>false</loop><delay_start>0.0</delay_start><auto_start>true</auto_start>
        <trajectory id="0" type="{nome_anim}">{''.join(pontos)}</trajectory>
      </script>
      <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>{rotulo}</label></plugin>
    </actor>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cenario", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--n_pessoas", type=int, default=None)
    ap.add_argument("--drones", default=None)
    a = ap.parse_args()
    cen_dir = Path(a.cenario)
    out = Path(a.saida)
    (out / "modelos" / "terreno").mkdir(parents=True, exist_ok=True)
    cen = json.loads((cen_dir / "cenario.json").read_text(encoding="utf-8"))
    cena = json.loads((cen_dir / "cena.json").read_text(encoding="utf-8"))
    pessoas = json.loads((cen_dir / "pessoas_plano.json").read_text(encoding="utf-8"))["pessoas"]
    if a.n_pessoas:
        pessoas = pessoas[: a.n_pessoas]
    if len(pessoas) > 254:
        raise SystemExit("no maximo 254 pessoas (rotulo do sistema Label e 1..255)")
    drones = [d for d in cen["drones"] if not a.drones or d["id"] in a.drones.split(",")]
    tex_dir = out / "modelos" / "terreno" / "texturas"
    if (cen_dir / "texturas").exists():
        shutil.copytree(cen_dir / "texturas", tex_dir, dirs_exist_ok=True)
    tex = cen.get("texturas", {})

    visuais = []
    for mm in cena["malhas"]:
        if not mm["vertices"]:
            continue
        nome = mm["nome"]
        obj(out / "modelos" / "terreno" / f"{nome}.obj", mm["vertices"], mm["faces"], mm["uv"])
        t = f"model://terreno/texturas/{mm['material']}.jpg" if mm["material"] in tex else None
        visuais.append(f"""
        <visual name="{nome}"><geometry><mesh><uri>model://terreno/{nome}.obj</uri></mesh></geometry>
          {material_pbr(t, rug=tex.get(mm['material'], {}).get('rugosidade', 0.8))}</visual>""")
    (out / "modelos" / "terreno" / "model.config").write_text(
        "<?xml version='1.0'?><model><name>terreno</name><version>1.0</version><sdf version='1.9'>model.sdf</sdf></model>", encoding="utf-8")
    x0, y0, x1, y1 = cena["chao_retangulo"]
    (out / "modelos" / "terreno" / "model.sdf").write_text(f"""<?xml version="1.0"?>
<sdf version="1.9"><model name="terreno"><static>true</static><link name="chao">
  <collision name="plano"><geometry><plane><normal>0 0 1</normal><size>{x1 - x0:.1f} {y1 - y0:.1f}</size></plane></geometry></collision>
  {''.join(visuais)}
</link></model></sdf>
""", encoding="utf-8")

    edif = []
    for i, e in enumerate(cena["edificios"]):
        P = np.array(e["pegada"])
        c = P.mean(axis=0)
        d01 = P[1] - P[0]; d12 = P[2] - P[1]
        w, h = np.linalg.norm(d01), np.linalg.norm(d12)
        yaw = math.atan2(d01[1], d01[0])
        H = e["altura_m"]
        edif.append(f"""
    <model name="edificio_{i:02d}"><static>true</static><pose>{c[0]:.2f} {c[1]:.2f} {H / 2:.2f} 0 0 {yaw:.4f}</pose>
      <link name="l"><collision name="c"><geometry><box><size>{w:.2f} {h:.2f} {H:.2f}</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>{w:.2f} {h:.2f} {H:.2f}</size></box></geometry>{material_pbr(None, (0.78, 0.75, 0.70), 0.8)}</visual></link></model>""")
    arv = []
    for i, t in enumerate(cena["arvores"]):
        ht = 0.45 * t["altura_m"]
        r = t["raio_copa_m"]
        arv.append(f"""
    <model name="arvore_{i:03d}"><static>true</static><pose>{t['x']:.2f} {t['y']:.2f} 0 0 0 0</pose><link name="l">
      <visual name="tronco"><pose>0 0 {ht / 2:.2f} 0 0 0</pose><geometry><cylinder><radius>{0.18 + 0.03 * r:.2f}</radius><length>{ht:.2f}</length></cylinder></geometry>{material_pbr(None, (0.30, 0.22, 0.14), 0.9)}</visual>
      <visual name="copa"><pose>0 0 {ht + 0.6 * r:.2f} 0 0 0</pose><geometry><ellipsoid><radii>{r:.2f} {r:.2f} {0.75 * r + 0.25 * (t['altura_m'] - ht):.2f}</radii></ellipsoid></geometry>{material_pbr(None, (0.16, 0.30, 0.12), 0.9)}</visual>
    </link></model>""")

    instancias = []
    inc_drones = []
    for i, d in enumerate(drones):
        K = Intrinsecos.de_dict(d["camera"]["intrinsecos"])
        modelo_drone(out / "modelos", d["id"], K, a.fps, d["camera"].get("inclinacao_graus", 0.0))
        bx, by = d["base_enu"]
        nome = f"x500_cerise_{d['id']}"
        inc_drones.append(f"""
    <include><uri>model://{nome}</uri><name>{nome}</name><pose>{bx:.2f} {by:.2f} 0.25 0 0 0</pose></include>""")
        instancias.append({"drone": d["id"], "instancia_px4": i + 1, "modelo_gz": nome, "porta_mavsdk": 14540 + i + 1,
                           "plan": str(Path(cen_dir.name) / d["plan_qgc"]), "base_enu": d["base_enu"],
                           "base_lonlat": d["base_lonlat"]})
    # camera de visao geral (estatica): o eixo X do sensor aponta para o alvo
    vg = cen["visao_geral"]
    Kv = Intrinsecos.de_dict(vg["intrinsecos"])
    dvx, dvy, dvz = (vg["alvo"][k] - vg["posicao"][k] for k in range(3))
    yaw_v = math.atan2(dvy, dvx)
    pitch_v = math.atan2(-dvz, math.hypot(dvx, dvy))
    cam_vg = f"""
    <model name="camera_visao_geral"><static>true</static>
      <pose>{vg['posicao'][0]:.2f} {vg['posicao'][1]:.2f} {vg['posicao'][2]:.2f} 0 {pitch_v:.5f} {yaw_v:.5f}</pose>
      <link name="l"><sensor name="cam" type="camera"><camera><horizontal_fov>{math.radians(Kv.hfov_graus):.5f}</horizontal_fov>
        <image><width>{Kv.largura}</width><height>{Kv.altura}</height><format>R8G8B8</format></image><clip><near>1</near><far>5000</far></clip></camera>
        <always_on>1</always_on><update_rate>{a.fps}</update_rate><topic>/cerise/visao_geral/camera</topic></sensor></link>
    </model>"""
    o = cen["origem_enu"]
    sol_el, sol_az = math.radians(55), math.radians(120)
    dsol = (-math.cos(sol_el) * math.cos(sol_az), -math.cos(sol_el) * math.sin(sol_az), -math.sin(sol_el))
    dur = cen["duracao_s"]
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="ufg">
    <physics name="1ms" type="ignored"><max_step_size>0.004</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"><dynamic_pose_hertz>100</dynamic_pose_hertz></plugin>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-air-pressure-system" name="gz::sim::systems::AirPressure"/>
    <plugin filename="gz-sim-magnetometer-system" name="gz::sim::systems::Magnetometer"/>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>
    <spherical_coordinates><surface_model>EARTH_WGS84</surface_model><world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>{o['lat0']}</latitude_deg><longitude_deg>{o['lon0']}</longitude_deg><elevation>0</elevation><heading_deg>0</heading_deg>
    </spherical_coordinates>
    <scene><ambient>0.55 0.58 0.62 1</ambient><background>0.70 0.80 0.95 1</background><shadows>true</shadows><grid>false</grid></scene>
    <light type="directional" name="sol"><cast_shadows>true</cast_shadows><pose>0 0 200 0 0 0</pose>
      <diffuse>1.0 0.96 0.9 1</diffuse><specular>0.3 0.3 0.3 1</specular><direction>{dsol[0]:.4f} {dsol[1]:.4f} {dsol[2]:.4f}</direction></light>
    <include><uri>model://terreno</uri><name>terreno</name><pose>0 0 0 0 0 0</pose></include>
    {''.join(edif)}
    {''.join(arv)}
    {''.join(ator(p, dur, k + 1) for k, p in enumerate(pessoas))}
    {''.join(inc_drones)}
    {cam_vg}
  </world>
</sdf>
"""
    (out / "mundo_ufg.sdf").write_text(sdf, encoding="utf-8")
    (out / "px4_instancias.json").write_text(json.dumps({"origem_enu": o, "fps": a.fps, "duracao_s": dur,
                                                         "pessoas": [p["id"] for p in pessoas],
                                                         "rotulos": {k + 1: p["id"] for k, p in enumerate(pessoas)},
                                                         "camera_no_modelo": {"t": [0, 0, -0.13], "R_modelo_cv": [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]},
                                                         "instancias": instancias}, indent=1),
                                             encoding="utf-8")
    print(f"mundo: {out / 'mundo_ufg.sdf'} ({len(pessoas)} atores, {len(drones)} drones, {len(edif)} edificios, {len(arv)} arvores)")


if __name__ == "__main__":
    main()
