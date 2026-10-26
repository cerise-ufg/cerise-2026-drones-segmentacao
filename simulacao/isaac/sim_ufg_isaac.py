"""Simulacao realista no Isaac Sim 5.1 (headless): drones percorrendo a area da UFG (rotas DARP+refinamento
do cerise_drones) com pessoas animadas (omni.anim.people), gravando por drone RGB + caixas 2D de pessoas
(Replicator bounding_box_2d_tight) + pose da camera a cada quadro, e uma camera de visao geral.

ESCRITO E NAO EXECUTADO nesta etapa (o aceite da licenca do Isaac Sim ainda nao foi confirmado).
Rodar dentro do conteiner oficial (ver isaac/rodar_isaac.sh):
    /isaac-sim/python.sh /sim/isaac/sim_ufg_isaac.py --cenario /cenario/reduzido --saida /saida/run01 --fps 5

Formato de saida: o de cerise_sim/dados.py (meta.json, gabarito_pessoas.csv, <cam>/rgb, <cam>/poses.csv,
<cam>/caixas_gt.csv), mais <cam>/camera_params.jsonl (anotador camera_params, para conferencia).

Modos de pessoas:
  anim_people (padrao): personagens de Isaac/People/Characters com grafo de animacao (Biped_Setup) e o
      script de comportamento do omni.anim.people seguindo pessoas_comandos.txt (GoTo/Idle/LookAround) sobre
      uma navmesh assada na area; o gabarito e a posicao REAL lida do simulador a cada quadro.
  cinematico (alternativa): mesmos personagens; a posicao/rumo vem do plano (set_world_transform) e o
      grafo recebe Action=Walk/None so para as pernas andarem. Gabarito tambem lido do simulador.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--cenario", required=True, help="pasta gerada por scripts/gerar_cenario.py")
ap.add_argument("--saida", required=True)
ap.add_argument("--duracao", type=float, default=None, help="s de simulacao a gravar (padrao: duracao do cenario)")
ap.add_argument("--t_inicio", type=float, default=0.0, help="instante do plano em que a gravacao comeca")
ap.add_argument("--fps", type=float, default=5.0, help="quadros gravados por segundo de simulacao")
ap.add_argument("--dt", type=float, default=1.0 / 30.0, help="passo de simulacao/animacao (s)")
ap.add_argument("--n_pessoas", type=int, default=None, help="usa as N primeiras pessoas do plano")
ap.add_argument("--pessoas", default=None, help="lista explicita de ids do plano, ex.: P000,P022 (smoke test)")
ap.add_argument("--largura", type=int, default=None, help="sobrescreve a largura (px) das cameras dos drones, mantendo FOV e aspecto")
ap.add_argument("--drones", default=None, help="subconjunto, ex.: D1,D3")
ap.add_argument("--modo_pessoas", choices=["anim_people", "cinematico"], default="anim_people")
ap.add_argument("--materiais", choices=["procedural", "biblioteca"], default="procedural")
ap.add_argument("--modelo_drone", choices=["procedural", "crazyflie_local", "quadcopter_s3"], default="procedural")
ap.add_argument("--escala_drone_visual", type=float, default=5.0, help="aumenta o modelo visual p/ aparecer na visao geral")
ap.add_argument("--ativos", default=os.environ.get("ISAAC_ASSETS_ROOT"),
                help="raiz dos ativos Isaac (padrao: servidor configurado na imagem; ou espelho local /ativos/Assets/Isaac/5.1)")
ap.add_argument("--sem_visao_geral", action="store_true")
ap.add_argument("--largura_visao_geral", type=int, default=1920)
ap.add_argument("--renderer", default="RaytracedLighting", choices=["RaytracedLighting", "PathTracing"])
ap.add_argument("--rt_subframes", type=int, default=2)
ap.add_argument("--formato", choices=["jpg", "png"], default="jpg")
ap.add_argument("--qualidade_jpg", type=int, default=93)
ap.add_argument("--sol_elevacao", type=float, default=55.0)
ap.add_argument("--sol_azimute", type=float, default=120.0, help="graus a partir do leste, anti-horario")
ap.add_argument("--sol_intensidade", type=float, default=3500.0)
ap.add_argument("--ceu_intensidade", type=float, default=900.0)
ap.add_argument("--aquecimento_s", type=float, default=3.0, help="s simulados antes de gravar (grafo de animacao/navmesh)")
ap.add_argument("--smoke", action="store_true", help="teste rapido: 1 drone, 6 pessoas, 4 s")
ap.add_argument("--camera_debug", default=None, help="id de pessoa: camera baixa que a segue (conferir animacao/orientacao)")
ap.add_argument("--yaw_offset", type=float, default=0.0, help="graus somados ao rumo do plano no modo cinematico")
ap.add_argument("--depurar", action="store_true", help="registra estado do grafo de animacao/navmesh das pessoas")
ap.add_argument("--max_quadros", type=int, default=None)
args, _desconhecidos = ap.parse_known_args()
if args.smoke:
    args.drones = args.drones or "D1"
    args.n_pessoas = args.n_pessoas or 6
    args.duracao = args.duracao or 4.0

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": args.renderer, "width": 1280, "height": 720,
                     "anti_aliasing": 0, "multi_gpu": False})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade  # noqa: E402

SIM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM_DIR))
from cerise_sim.camera import Intrinsecos, olhar_para, pose_drone, quat_de_matriz, r_usd_de_cv  # noqa: E402
from cerise_sim.dados import CAB_CAIXAS, CAB_GAB, CAB_POSES  # noqa: E402
from cerise_sim.trajetoria import Trajetoria  # noqa: E402

LOG = []


def log(msg):
    s = f"[cerise_sim {time.strftime('%H:%M:%S')}] {msg}"
    print(s, flush=True)
    LOG.append(s)


for ext in ["omni.replicator.core", "omni.kit.scripting", "omni.anim.timeline", "omni.anim.graph.bundle",
            "omni.anim.graph.core", "omni.anim.retarget.core", "omni.anim.navigation.bundle", "omni.anim.navigation.core",
            "omni.anim.people", "isaacsim.core.utils"]:
    try:
        enable_extension(ext)
        app.update()
    except Exception as e:  # noqa: BLE001
        log(f"AVISO: extensao {ext} nao habilitada: {e!r}")

import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402

carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
carb.settings.get_settings().set("/app/runLoops/main/rateLimitEnabled", False)

# ---------------------------------------------------------------------------------------------- cenario
CEN = Path(args.cenario)
cen = json.loads((CEN / "cenario.json").read_text(encoding="utf-8"))
cena = json.loads((CEN / "cena.json").read_text(encoding="utf-8"))
plano_p = json.loads((CEN / "pessoas_plano.json").read_text(encoding="utf-8"))["pessoas"]
if args.pessoas:
    _sel = args.pessoas.split(",")
    plano_p = [p for p in plano_p if p["id"] in _sel]
elif args.n_pessoas:
    plano_p = plano_p[: args.n_pessoas]
drones = [d for d in cen["drones"] if not args.drones or d["id"] in args.drones.split(",")]
duracao = min(args.duracao or cen["duracao_s"], cen["duracao_s"] - args.t_inicio)
OUT = Path(args.saida)
OUT.mkdir(parents=True, exist_ok=True)


def raiz_ativos():
    if args.ativos:
        return args.ativos.rstrip("/")
    try:
        from isaacsim.storage.native import get_assets_root_path
        r = get_assets_root_path()
        if r:
            return r.rstrip("/")
    except Exception as e:  # noqa: BLE001
        log(f"AVISO get_assets_root_path: {e!r}")
    return os.environ.get("OMNI_SERVER", "").rstrip("/")


ATIVOS = raiz_ativos()
log(f"raiz de ativos: {ATIVOS!r}; cenario {cen['nome']}; {len(drones)} drones; {len(plano_p)} pessoas; "
    f"{duracao:.1f} s a {args.fps} fps; modo {args.modo_pessoas}")

# ---------------------------------------------------------------------------------------------- palco
ctx = omni.usd.get_context()
ctx.new_stage()
app.update()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
for p in ["/World", "/World/Terreno", "/World/Edificios", "/World/Arvores", "/World/Drones", "/World/Cameras", "/World/Looks"]:
    UsdGeom.Xform.Define(stage, p)
stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))


def material_omnipbr(nome, textura=None, cor=(0.5, 0.5, 0.5), rugosidade=0.8, mdl_url=None, mdl_nome=None):
    """OmniPBR (MDL local da imagem) com textura procedural; ou material MDL da biblioteca (S3)."""
    caminho = f"/World/Looks/{nome}"
    if mdl_url:
        omni.kit.commands.execute("CreateMdlMaterialPrimCommand", mtl_url=mdl_url, mtl_name=mdl_nome, mtl_path=caminho)
        return UsdShade.Material(stage.GetPrimAtPath(caminho))
    omni.kit.commands.execute("CreateMdlMaterialPrimCommand", mtl_url="OmniPBR.mdl", mtl_name="OmniPBR", mtl_path=caminho)
    sh = UsdShade.Shader(stage.GetPrimAtPath(caminho + "/Shader"))
    if textura:
        sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(str(textura))
    sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*cor))
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(rugosidade))
    return UsdShade.Material(stage.GetPrimAtPath(caminho))


BIBLIOTECA = {  # materiais MDL da biblioteca NVIDIA (servidor de ativos), usados com --materiais biblioteca
    "grama": ("NVIDIA/Materials/Base/Natural/Grass_Countryside.mdl", "Grass_Countryside"),
    "grama_seca": ("NVIDIA/Materials/Base/Natural/Grass_Cut.mdl", "Grass_Cut"),
    "terra": ("NVIDIA/Materials/Base/Natural/Dirt.mdl", "Dirt"),
    "asfalto": ("NVIDIA/Materials/Base/Natural/Asphalt.mdl", "Asphalt"),
    "concreto": ("NVIDIA/Materials/vMaterials_2/Concrete/Concrete_Rough.mdl", "Concrete_Rough"),
}
MATS = {}
raiz_servidor = ATIVOS.split("/Isaac/")[0] if "/Isaac/" in ATIVOS else ATIVOS
for nome, info in cen.get("texturas", {}).items():
    tex = (CEN / info["arquivo"]).resolve()
    if args.materiais == "biblioteca" and nome in BIBLIOTECA:
        url, mn = BIBLIOTECA[nome]
        try:
            MATS[nome] = material_omnipbr(nome, mdl_url=f"{raiz_servidor}/{url}", mdl_nome=mn)
            continue
        except Exception as e:  # noqa: BLE001
            log(f"AVISO material da biblioteca {nome} falhou ({e!r}); usando procedural")
    MATS[nome] = material_omnipbr(nome, tex, rugosidade=info.get("rugosidade", 0.8))
MATS.setdefault("tronco", material_omnipbr("tronco", cor=(0.30, 0.22, 0.14), rugosidade=0.9))
MATS.setdefault("copa", material_omnipbr("copa", cor=(0.16, 0.30, 0.12), rugosidade=0.9))


def malha(caminho, V, F, uv=None, material=None):
    m = UsdGeom.Mesh.Define(stage, caminho)
    m.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in V])
    m.CreateFaceVertexCountsAttr([len(f) for f in F])
    m.CreateFaceVertexIndicesAttr([int(i) for f in F for i in f])
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(False)
    if uv is not None:
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set([Gf.Vec2f(float(a), float(b)) for a, b in uv])
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    return m


t0 = time.time()
for i, mm in enumerate(cena["malhas"]):
    if mm["vertices"]:
        malha(f"/World/Terreno/{mm['nome']}", mm["vertices"], mm["faces"], mm["uv"], MATS.get(mm["material"]))
mpl_parede = cen.get("texturas", {}).get("parede", {}).get("metros_por_ladrilho", 6.0)
mpl_telhado = cen.get("texturas", {}).get("telhado", {}).get("metros_por_ladrilho", 8.0)
for i, e in enumerate(cena["edificios"]):
    mm = e["malha"]
    V = np.array(mm["vertices"]); F = mm["faces"]; UV = np.array(mm["uv_metros"])
    npf = mm["faces_parede"]
    idx_par = sorted({k for f in F[:npf] for k in f}); idx_tel = sorted({k for f in F[npf:] for k in f})
    for nome, idx, faces, mat, esc in [("paredes", idx_par, F[:npf], "parede", mpl_parede), ("telhado", idx_tel, F[npf:], "telhado", mpl_telhado)]:
        mapa = {k: j for j, k in enumerate(idx)}
        malha(f"/World/Edificios/E{i:02d}_{nome}", V[idx], [[mapa[k] for k in f] for f in faces], UV[idx] / esc, MATS.get(mat))
for i, a in enumerate(cena["arvores"]):
    base = f"/World/Arvores/A{i:03d}"
    UsdGeom.Xform.Define(stage, base)
    h_tronco = a["altura_m"] * 0.45
    tr = UsdGeom.Cylinder.Define(stage, base + "/tronco")
    tr.CreateRadiusAttr(0.18 + 0.03 * a["raio_copa_m"]); tr.CreateHeightAttr(h_tronco); tr.CreateAxisAttr("Z")
    UsdGeom.XformCommonAPI(tr).SetTranslate(Gf.Vec3d(a["x"], a["y"], h_tronco / 2))
    UsdShade.MaterialBindingAPI.Apply(tr.GetPrim()).Bind(MATS["tronco"])
    cp = UsdGeom.Sphere.Define(stage, base + "/copa")
    cp.CreateRadiusAttr(1.0)
    UsdGeom.XformCommonAPI(cp).SetTranslate(Gf.Vec3d(a["x"], a["y"], h_tronco + 0.6 * a["raio_copa_m"]))
    UsdGeom.XformCommonAPI(cp).SetScale(Gf.Vec3f(a["raio_copa_m"], a["raio_copa_m"], 0.75 * a["raio_copa_m"] + 0.25 * (a["altura_m"] - h_tronco)))
    UsdShade.MaterialBindingAPI.Apply(cp.GetPrim()).Bind(MATS["copa"])
log(f"terreno: {len(cena['malhas'])} malhas, {len(cena['edificios'])} edificios, {len(cena['arvores'])} arvores ({time.time() - t0:.1f} s)")

# luz: sol (DistantLight) + ceu (DomeLight)
sol = UsdLux.DistantLight.Define(stage, "/World/Sol")
sol.CreateIntensityAttr(args.sol_intensidade); sol.CreateAngleAttr(0.53); sol.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
el, az = math.radians(args.sol_elevacao), math.radians(args.sol_azimute)
d_sol = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])  # direcao PARA o sol
R_sol = olhar_para(d_sol * 1000.0, np.zeros(3))  # luz distante emite ao longo de -Z local (como a camera USD)
q = quat_de_matriz(r_usd_de_cv(R_sol))
UsdGeom.Xformable(sol).AddOrientOp().Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
ceu = UsdLux.DomeLight.Define(stage, "/World/Ceu")
ceu.CreateIntensityAttr(args.ceu_intensidade); ceu.CreateColorAttr(Gf.Vec3f(0.78, 0.86, 1.0))


# ---------------------------------------------------------------------------------------------- drones
def drone_procedural(caminho, cor, escala):
    UsdGeom.Xform.Define(stage, caminho)
    s = 0.25 * escala
    corpo = UsdGeom.Cube.Define(stage, caminho + "/corpo"); corpo.CreateSizeAttr(1.0)
    UsdGeom.XformCommonAPI(corpo).SetScale(Gf.Vec3f(0.6 * s, 0.6 * s, 0.25 * s))
    corpo.CreateDisplayColorAttr([Gf.Vec3f(*cor)])
    for k, ang in enumerate((45, 135, 225, 315)):
        a = math.radians(ang)
        braco = UsdGeom.Cube.Define(stage, f"{caminho}/braco{k}"); braco.CreateSizeAttr(1.0)
        UsdGeom.XformCommonAPI(braco).SetTranslate(Gf.Vec3d(0.6 * s * math.cos(a), 0.6 * s * math.sin(a), 0))
        UsdGeom.XformCommonAPI(braco).SetRotate(Gf.Vec3f(0, 0, ang))
        UsdGeom.XformCommonAPI(braco).SetScale(Gf.Vec3f(1.2 * s, 0.08 * s, 0.06 * s))
        braco.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.1, 0.1)])
        rot = UsdGeom.Cylinder.Define(stage, f"{caminho}/rotor{k}")
        rot.CreateRadiusAttr(0.45 * s); rot.CreateHeightAttr(0.02 * s); rot.CreateAxisAttr("Z")
        UsdGeom.XformCommonAPI(rot).SetTranslate(Gf.Vec3d(1.2 * s * math.cos(a), 1.2 * s * math.sin(a), 0.08 * s))
        rot.CreateDisplayColorAttr([Gf.Vec3f(*cor)])


# escala aplicada ao Xform do modelo visual (o procedural ja nasce com a escala na geometria)
ESCALA_VIS = {"procedural": 1.0, "crazyflie_local": 10.0 * args.escala_drone_visual, "quadcopter_s3": args.escala_drone_visual}[args.modelo_drone]
CORES = [(0.12, 0.47, 0.71), (0.84, 0.15, 0.16), (0.17, 0.63, 0.17), (0.58, 0.40, 0.74), (1.0, 0.5, 0.05)]
TRAJ, KS, CAMS, VIS, ANNOT, RPS = {}, {}, {}, {}, {}, []
for i, d in enumerate(drones):
    did = d["id"]
    TRAJ[did] = Trajetoria.de_dict(d["trajetoria"])
    K = Intrinsecos.de_dict(d["camera"]["intrinsecos"])
    if args.largura:
        K = Intrinsecos.de_hfov(args.largura, int(round(args.largura * K.altura / K.largura / 2)) * 2, K.hfov_graus)
    KS[did] = (K, d["camera"].get("inclinacao_graus", 0.0))
    vis = f"/World/Drones/{did}"
    try:
        if args.modelo_drone == "crazyflie_local":
            cf = glob.glob("/isaac-sim/extscache/omni.warp.core-*/warp/examples/assets/crazyflie.usd")
            UsdGeom.Xform.Define(stage, vis).GetPrim().GetReferences().AddReference(cf[0])

        elif args.modelo_drone == "quadcopter_s3":
            UsdGeom.Xform.Define(stage, vis).GetPrim().GetReferences().AddReference(f"{ATIVOS}/Isaac/Robots/IsaacSim/Quadcopter/quadcopter.usd")
        else:
            drone_procedural(vis, CORES[i % len(CORES)], args.escala_drone_visual)
    except Exception as e:  # noqa: BLE001
        log(f"AVISO modelo do drone {did}: {e!r}; usando procedural")
        drone_procedural(vis, CORES[i % len(CORES)], args.escala_drone_visual)
    VIS[did] = UsdGeom.Xformable(stage.GetPrimAtPath(vis))
    # o modelo visual (ampliado p/ a visao geral) nao projeta sombra: a sombra ampliada apareceria nas cameras
    from pxr import Usd as _Usd
    for _p in _Usd.PrimRange(stage.GetPrimAtPath(vis)):
        if _p.IsA(UsdGeom.Gprim):
            UsdGeom.PrimvarsAPI(_p).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    cam = UsdGeom.Camera.Define(stage, f"/World/Cameras/{did}")
    u = K.usd()
    cam.CreateFocalLengthAttr(u["focalLength"]); cam.CreateHorizontalApertureAttr(u["horizontalAperture"])
    cam.CreateVerticalApertureAttr(u["verticalAperture"]); cam.CreateClippingRangeAttr(Gf.Vec2f(0.5, 5000.0))
    CAMS[did] = cam

if not args.sem_visao_geral:
    vg = cen["visao_geral"]
    Kv = Intrinsecos.de_hfov(args.largura_visao_geral, int(args.largura_visao_geral * 9 / 16), Intrinsecos.de_dict(vg["intrinsecos"]).hfov_graus)
    KS["visao_geral"] = (Kv, None)
    cam = UsdGeom.Camera.Define(stage, "/World/Cameras/visao_geral")
    u = Kv.usd()
    cam.CreateFocalLengthAttr(u["focalLength"]); cam.CreateHorizontalApertureAttr(u["horizontalAperture"])
    cam.CreateVerticalApertureAttr(u["verticalAperture"]); cam.CreateClippingRangeAttr(Gf.Vec2f(1.0, 10000.0))
    CAMS["visao_geral"] = cam

XOPS = {}


def por_pose(nome, prim_xf, C, R_usd):
    if nome not in XOPS:
        prim_xf.ClearXformOpOrder()
        XOPS[nome] = (prim_xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble), prim_xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble))
    q = quat_de_matriz(R_usd)
    XOPS[nome][0].Set(Gf.Vec3d(*map(float, C)))
    XOPS[nome][1].Set(Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3])))


if "visao_geral" in CAMS:
    Rv = olhar_para(cen["visao_geral"]["posicao"], cen["visao_geral"]["alvo"])
    por_pose("visao_geral", UsdGeom.Xformable(CAMS["visao_geral"]), cen["visao_geral"]["posicao"], r_usd_de_cv(Rv))

P_DEBUG = None
if args.camera_debug:
    P_DEBUG = next((p for p in plano_p if p["id"] == args.camera_debug), None)
    if P_DEBUG is not None:
        Kd = Intrinsecos.de_hfov(960, 540, 60.0)
        KS["debug"] = (Kd, None)
        cam = UsdGeom.Camera.Define(stage, "/World/Cameras/debug")
        u = Kd.usd()
        cam.CreateFocalLengthAttr(u["focalLength"]); cam.CreateHorizontalApertureAttr(u["horizontalAperture"])
        cam.CreateVerticalApertureAttr(u["verticalAperture"]); cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 2000.0))
        CAMS["debug"] = cam
DEBUG_POSE = {}


def atualizar_drones(t):
    estado = {}
    for did, tr in TRAJ.items():
        x, y, z, yaw, cobre, _f = tr.amostrar(args.t_inicio + t)
        K, incl = KS[did]
        pose = pose_drone(float(x[0]), float(y[0]), max(float(z[0]), 0.3), float(yaw[0]), incl)
        por_pose(did, UsdGeom.Xformable(CAMS[did]), pose.C, r_usd_de_cv(pose.R_wc))
        # corpo do drone 6 cm acima do centro optico, rumo = yaw
        cy, sy = math.cos(math.radians(yaw[0]) / 2), math.sin(math.radians(yaw[0]) / 2)
        nm = did + "_vis"
        if nm not in XOPS:
            VIS[did].ClearXformOpOrder()
            XOPS[nm] = (VIS[did].AddTranslateOp(UsdGeom.XformOp.PrecisionDouble), VIS[did].AddOrientOp(UsdGeom.XformOp.PrecisionDouble),
                        VIS[did].AddScaleOp())
            XOPS[nm][2].Set(Gf.Vec3f(ESCALA_VIS))
        XOPS[nm][0].Set(Gf.Vec3d(float(x[0]), float(y[0]), max(float(z[0]), 0.3) + 0.06 * args.escala_drone_visual))
        XOPS[nm][1].Set(Gf.Quatd(cy, 0.0, 0.0, sy))
        estado[did] = (pose, float(yaw[0]), bool(cobre[0]))
    return estado


# ---------------------------------------------------------------------------------------------- pessoas
PERS = {}            # id -> caminho do prim raiz
SKEL = {}            # id -> caminho do SkelRoot
AVISOS = []


def listar_personagens():
    import omni.client
    base = f"{ATIVOS}/Isaac/People/Characters"
    res, itens = omni.client.list(base + "/")
    if res != omni.client.Result.OK:
        raise RuntimeError(f"nao consegui listar {base}: {res}")
    pastas = sorted(i.relative_path.rstrip("/") for i in itens if (i.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN)
                    and not i.relative_path.startswith(".") and not i.relative_path.lower().startswith("biped")
                    and not i.relative_path.startswith("DH_"))
    usds = []
    for p in pastas:
        r2, fs = omni.client.list(f"{base}/{p}/")
        u = sorted(f.relative_path for f in fs if f.relative_path.endswith((".usd", ".usda")) and not f.relative_path.startswith("."))
        if not u:
            continue
        # prefere <pasta>.usd (personagem completo) a variantes
        pref = [x for x in u if x.rsplit(".", 1)[0] == p]
        usds.append(f"{base}/{p}/{(pref or u)[0]}")
    return usds


def rotulo_pessoa(prim):
    try:
        from isaacsim.core.utils.semantics import add_labels
        add_labels(prim, labels=["person"], instance_name="class")
    except Exception:  # noqa: BLE001
        from isaacsim.core.utils.semantics import add_update_semantics
        add_update_semantics(prim, "person")


def criar_pessoas():
    from pxr import Usd
    UsdGeom.Xform.Define(stage, "/World/Characters")
    usds = listar_personagens()
    log(f"{len(usds)} modelos de personagem disponiveis")
    for k, p in enumerate(plano_p):
        caminho = f"/World/Characters/{p['id']}"
        xf = UsdGeom.Xform.Define(stage, caminho)
        xf.GetPrim().GetReferences().AddReference(usds[k % len(usds)])
        _t, x, y, yaw = p["kf"][0]
        xf.ClearXformOpOrder()
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(x, y, 0.0))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(math.cos(math.radians(yaw) / 2), 0, 0, math.sin(math.radians(yaw) / 2)))
        rotulo_pessoa(xf.GetPrim())
        PERS[p["id"]] = caminho
    app.update()
    for pid, caminho in PERS.items():
        for prim in Usd.PrimRange(stage.GetPrimAtPath(caminho)):
            if prim.GetTypeName() == "SkelRoot":
                SKEL[pid] = str(prim.GetPath())
                break
    try:
        import NavSchema
        for caminho in PERS.values():
            omni.kit.commands.execute("ApplyNavMeshAPICommand", prim_path=caminho, api=NavSchema.NavMeshExcludeAPI)
    except Exception as e:  # noqa: BLE001
        log(f"AVISO NavMeshExcludeAPI: {e!r}")
    # Biped_Setup (grafo de animacao) + script de comportamento do omni.anim.people
    biped = UsdGeom.Xform.Define(stage, "/World/Characters/Biped_Setup")
    biped.GetPrim().GetReferences().AddReference(f"{ATIVOS}/Isaac/People/Characters/Biped_Setup.usd")
    UsdGeom.Imageable(biped).MakeInvisible()
    app.update()
    try:
        from omni.anim.people.scripts.custom_command.populate_anim_graph import populate_anim_graph
        populate_anim_graph()
    except Exception as e:  # noqa: BLE001
        log(f"AVISO populate_anim_graph: {e!r}")
    grafo = None
    for prim in Usd.PrimRange(biped.GetPrim()):
        if prim.GetTypeName() == "AnimationGraph":
            grafo = prim.GetPath()
            break
    if grafo is None:
        raise RuntimeError("AnimationGraph nao encontrado no Biped_Setup (ativos inacessiveis?)")
    caminhos = [Sdf.Path(p) for p in SKEL.values()]
    omni.kit.commands.execute("RemoveAnimationGraphAPICommand", paths=caminhos)
    omni.kit.commands.execute("ApplyAnimationGraphAPICommand", paths=caminhos, animation_graph_path=grafo)
    if args.modo_pessoas == "anim_people":
        mgr = omni.kit.app.get_app().get_extension_manager()
        script = mgr.get_extension_path_by_module("omni.anim.people") + "/omni/anim/people/scripts/character_behavior.py"
        omni.kit.commands.execute("RemoveScriptingAPICommand", paths=caminhos)
        omni.kit.commands.execute("ApplyScriptingAPICommand", paths=caminhos)
        for pth in SKEL.values():
            stage.GetPrimAtPath(pth).GetAttribute("omni:scripting:scripts").Set([script])
        # arquivo de comandos so com as pessoas usadas
        cmd = OUT / "pessoas_comandos_usados.txt"
        ids = set(PERS)
        linhas = [ln for ln in (CEN / "pessoas_comandos.txt").read_text(encoding="utf-8").splitlines() if ln.split(" ")[0] in ids]
        cmd.write_text("\n".join(linhas) + "\n", encoding="utf-8")
        s = carb.settings.get_settings()
        s.set("/exts/omni.anim.people/command_settings/command_file_path", str(cmd))
        s.set("/persistent/exts/omni.anim.people/character_prim_path", "/World/Characters")
        s.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
        s.set("/exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled", True)
    log(f"pessoas criadas: {len(PERS)} (SkelRoots: {len(SKEL)}); grafo {grafo}")


def assar_navmesh():
    import omni.anim.navigation.core as nav
    omni.kit.commands.execute("CreateNavMeshVolumeCommand", parent_prim_path=Sdf.Path("/World"), volume_type=0)
    app.update()
    vol = None
    for prim in stage.Traverse():
        if prim.GetTypeName() == "NavMeshVolume":
            vol = prim
            break
    if vol is None:
        raise RuntimeError("NavMeshVolume nao criado")
    xs = [v[0] for mm in cena["malhas"] if mm["nome"] == "gramado" for v in mm["vertices"]]
    ys = [v[1] for mm in cena["malhas"] if mm["nome"] == "gramado" for v in mm["vertices"]]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    sx, sy = (max(xs) - min(xs)) / 2 + 20, (max(ys) - min(ys)) / 2 + 20
    # (o TransformPrim sobre o volume padrao nao teve efeito no 5.1: a navmesh ficava com +-4,5 m em torno da
    # origem.) Define direto a extensao local e as xformOps do volume.
    UsdGeom.Boundable(vol).GetExtentAttr().Set([Gf.Vec3f(-sx, -sy, -3.0), Gf.Vec3f(sx, sy, 3.0)])
    xf = UsdGeom.Xformable(vol)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(cx, cy, 1.0))
    app.update()
    bb = UsdGeom.Imageable(vol).ComputeWorldBound(0, UsdGeom.Tokens.default_).ComputeAlignedRange()
    t = time.time()
    inav = nav.acquire_interface()
    inav.start_navmesh_baking_and_wait()
    nm = inav.get_navmesh()
    ok = nm is not None
    # conferencia: caminho entre dois pontos afastados do gramado
    teste = None
    if ok:
        a, b = carb.Float3(cx - 0.3 * sx, cy, 0.0), carb.Float3(cx + 0.3 * sx, cy + 0.3 * sy, 0.0)
        r = nm.query_shortest_path(a, b, agent_radius=0.5)
        if r is not None:
            pts = r.get_points()
            teste = (len(pts), round(math.hypot(pts[-1][0] - b[0], pts[-1][1] - b[1]), 2))
    log(f"navmesh assada: {ok} ({time.time() - t:.1f} s), volume {vol.GetPath()} ~{2 * sx:.0f} x {2 * sy:.0f} m, "
        f"limites {tuple(round(v, 1) for v in bb.GetMin())}-{tuple(round(v, 1) for v in bb.GetMax())}; "
        f"caminho de teste (pontos, erro ao alvo m) = {teste}")
    if not ok or teste is None or teste[1] > 2.0:
        raise RuntimeError(f"navmesh nao cobre a area (teste {teste})")


modo_efetivo = args.modo_pessoas
try:
    criar_pessoas()
    if args.modo_pessoas == "anim_people":
        assar_navmesh()
except Exception as e:  # noqa: BLE001
    log("ERRO na configuracao das pessoas: " + repr(e))
    log(traceback.format_exc())
    if args.modo_pessoas == "anim_people" and PERS and SKEL:
        AVISOS.append(f"anim_people falhou ({e!r}); caindo para cinematico")
        modo_efetivo = "cinematico"
    else:
        raise


def plano_pose(p, t):
    kf = np.asarray(p["kf"], float)
    tt = min(max(t, kf[0, 0]), kf[-1, 0])
    x = np.interp(tt, kf[:, 0], kf[:, 1]); y = np.interp(tt, kf[:, 0], kf[:, 2]); w = np.interp(tt, kf[:, 0], kf[:, 3])
    tt2 = min(tt + 0.2, kf[-1, 0])
    v = math.hypot(np.interp(tt2, kf[:, 0], kf[:, 1]) - x, np.interp(tt2, kf[:, 0], kf[:, 2]) - y) / max(tt2 - tt, 1e-6)
    return float(x), float(y), float(w), v


_AG = {}


def personagem(pid):
    if pid not in _AG:
        try:
            import omni.anim.graph.core as ag
            _AG[pid] = ag.get_character(SKEL[pid])
        except Exception:  # noqa: BLE001
            _AG[pid] = None
    return _AG[pid]


def mover_cinematico(t):
    for p in plano_p:
        c = personagem(p["id"])
        if c is None:
            continue
        x, y, w, v = plano_pose(p, args.t_inicio + t)
        r = math.radians(w)
        try:
            ro = math.radians(w + args.yaw_offset)
            c.set_world_transform(carb.Float3(x, y, 0.0), carb.Float4(0.0, 0.0, math.sin(ro / 2), math.cos(ro / 2)))
            andando = v > 0.2
            c.set_variable("Action", "Walk" if andando else "None")
            c.set_variable("Walk", min(1.0, v / 1.4) if andando else 0.0)
        except Exception as e:  # noqa: BLE001
            if len(AVISOS) < 20:
                AVISOS.append(f"set_world_transform {p['id']}: {e!r}")


FONTE_GAB = {}


def depurar_pessoas():
    """Diagnostico do omni.anim.people: transformada, variaveis do grafo e caminho na navmesh."""
    try:
        import omni.anim.navigation.core as nav
        nm = nav.acquire_interface().get_navmesh()
    except Exception as e:  # noqa: BLE001
        nm = None
        log(f"DEPURAR navmesh: {e!r}")
    for p in plano_p[:6]:
        c = personagem(p["id"])
        if c is None:
            log(f"DEPURAR {p['id']} sem personagem"); continue
        pos = carb.Float3(0, 0, 0); rot = carb.Float4(0, 0, 0, 0)
        c.get_world_transform(pos, rot)
        info = {}
        for var in ("Action", "Walk", "PathPoints"):
            try:
                info[var] = str(c.get_variable(var))[:160]
            except Exception as e:  # noqa: BLE001
                info[var] = repr(e)[:80]
        alvo = next((cmd for cmd in p["comandos"] if cmd.startswith("GoTo")), None)
        cam = None
        if alvo and nm is not None:
            xs = alvo.split()
            try:
                r = nm.query_shortest_path(carb.Float3(pos.x, pos.y, pos.z), carb.Float3(float(xs[1]), float(xs[2]), float(xs[3])), agent_radius=0.5)
                cam = None if r is None else [(round(a[0], 1), round(a[1], 1), round(a[2], 1)) for a in r.get_points()][:5]
            except Exception as e:  # noqa: BLE001
                cam = repr(e)[:120]
        prim = stage.GetPrimAtPath(SKEL[p["id"]])
        mu = omni.usd.get_world_transform_matrix(prim).ExtractTranslation()
        log(f"DEPURAR {p['id']} grafo=({pos.x:.2f},{pos.y:.2f},{pos.z:.2f}) q=({rot.x:.3f},{rot.y:.3f},{rot.z:.3f},{rot.w:.3f}) "
            f"usd=({mu[0]:.2f},{mu[1]:.2f}) {info} alvo={alvo} caminho={cam}")


def posicoes_reais():
    out = {}
    for pid in PERS:
        c = personagem(pid)
        try:
            if c is not None:
                pos = carb.Float3(0, 0, 0); rot = carb.Float4(0, 0, 0, 0)
                c.get_world_transform(pos, rot)
                if abs(pos.x) + abs(pos.y) > 1e-6 or abs(rot.w) + abs(rot.z) > 1e-6:
                    yaw = math.degrees(2 * math.atan2(rot.z, rot.w))
                    out[pid] = (pos.x, pos.y, pos.z, yaw)
                    FONTE_GAB["anim_graph"] = FONTE_GAB.get("anim_graph", 0) + 1
                    continue
        except Exception:  # noqa: BLE001
            pass
        m = omni.usd.get_world_transform_matrix(stage.GetPrimAtPath(SKEL.get(pid, PERS[pid])))
        tr = m.ExtractTranslation()
        out[pid] = (tr[0], tr[1], tr[2], float("nan"))
        FONTE_GAB["usd"] = FONTE_GAB.get("usd", 0) + 1
    return out


# ---------------------------------------------------------------------------------------------- replicator
for cid, cam in CAMS.items():
    K, _ = KS[cid]
    rp = rep.create.render_product(str(cam.GetPath()), (K.largura, K.altura), name=f"rp_{cid}")
    RPS.append(rp)
    a_rgb = rep.AnnotatorRegistry.get_annotator("rgb"); a_rgb.attach(rp)
    a_par = rep.AnnotatorRegistry.get_annotator("camera_params"); a_par.attach(rp)
    a_bb = None
    if cid not in ("visao_geral", "debug"):
        a_bb = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight", init_params={"semanticTypes": ["class"]})
        a_bb.attach(rp)
    ANNOT[cid] = (rp, a_rgb, a_bb, a_par)
    (OUT / cid / "rgb").mkdir(parents=True, exist_ok=True)


def rps_ativos(on: bool):
    for rp in RPS:
        try:
            rp.hydra_texture.set_updates_enabled(on)
        except Exception:  # noqa: BLE001
            pass


GRAVADOR = ThreadPoolExecutor(max_workers=3)


def salvar_img(arr, caminho):
    from PIL import Image
    img = Image.fromarray(np.ascontiguousarray(arr[..., :3]))
    if args.formato == "jpg":
        img.save(caminho, quality=args.qualidade_jpg)
    else:
        img.save(caminho)


def jsonavel(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        a = np.asarray(v)
        return a.tolist() if a.dtype.kind in "biuf" else str(v)
    except Exception:  # noqa: BLE001
        return str(v)


def pid_de_caminho(caminho):
    partes = str(caminho).split("/")
    if "Characters" in partes:
        i = partes.index("Characters")
        if i + 1 < len(partes):
            return partes[i + 1]
    return None


# ---------------------------------------------------------------------------------------------- laco
from isaacsim.core.api import World  # noqa: E402

world = World(stage_units_in_meters=1.0, physics_dt=args.dt, rendering_dt=args.dt)
world.reset()
tl = omni.timeline.get_timeline_interface()
world.play()
rps_ativos(False)
n_aq = int(round(args.aquecimento_s / args.dt))
log(f"aquecimento: {n_aq} passos")
for _ in range(n_aq):
    if modo_efetivo == "cinematico":
        mover_cinematico(0.0)
    world.step(render=True)

passo_cap = max(1, int(round(1.0 / (args.fps * args.dt))))
n_passos = int(round(duracao / args.dt))
arq_poses = {c: open(OUT / c / "poses.csv", "w", encoding="utf-8") for c in CAMS}
arq_cx = {c: open(OUT / c / "caixas_gt.csv", "w", encoding="utf-8") for c in CAMS if c not in ("visao_geral", "debug")}
arq_par = {c: open(OUT / c / "camera_params.jsonl", "w", encoding="utf-8") for c in CAMS}
arq_gab = open(OUT / "gabarito_pessoas.csv", "w", encoding="utf-8")
for f in arq_poses.values():
    f.write(",".join(CAB_POSES) + "\n")
for f in arq_cx.values():
    f.write(",".join(CAB_CAIXAS) + "\n")
arq_gab.write(",".join(CAB_GAB) + "\n")

quadro = 0
t_ini_laco = time.time()
t_render = 0.0
for k in range(n_passos + 1):
    t = k * args.dt
    if modo_efetivo == "cinematico":
        mover_cinematico(t)
    if k % passo_cap == 0:
        estado = atualizar_drones(t)
        if P_DEBUG is not None:
            xd, yd, wd, _v = plano_pose(P_DEBUG, args.t_inicio + t)
            cdbg = personagem(P_DEBUG["id"])
            if cdbg is not None:
                _p = carb.Float3(0, 0, 0); _r = carb.Float4(0, 0, 0, 0)
                cdbg.get_world_transform(_p, _r)
                if abs(_p.x) + abs(_p.y) > 1e-6:
                    xd, yd = _p.x, _p.y
            rd = math.radians(wd)
            Cd = np.array([xd - 7 * math.cos(rd) + 2.5 * math.sin(rd), yd - 7 * math.sin(rd) - 2.5 * math.cos(rd), 3.5])
            Rd = olhar_para(Cd, np.array([xd, yd, 0.9]))
            por_pose("debug", UsdGeom.Xformable(CAMS["debug"]), Cd, r_usd_de_cv(Rd))
            DEBUG_POSE["C"], DEBUG_POSE["R"] = Cd, Rd
        rps_ativos(True)
        tr0 = time.time()
        rep.orchestrator.step(rt_subframes=args.rt_subframes, delta_time=0.0, pause_timeline=False)
        t_render += time.time() - tr0
        reais = posicoes_reais()
        for pid, (x, y, z, w) in reais.items():
            arq_gab.write(f"{quadro},{t:.4f},{pid},{x:.3f},{y:.3f},{z:.3f},{w:.2f}\n")
        for cid, (rp, a_rgb, a_bb, a_par) in ANNOT.items():
            rgb = a_rgb.get_data()
            if rgb is not None and getattr(rgb, "size", 0):
                GRAVADOR.submit(salvar_img, np.array(rgb), OUT / cid / "rgb" / f"{quadro:06d}.{args.formato}")
            par = a_par.get_data() or {}
            arq_par[cid].write(json.dumps({"frame": quadro, "t": t, **{kk: jsonavel(v) for kk, v in par.items()}}) + "\n")
            if cid == "debug":
                qd = quat_de_matriz(DEBUG_POSE["R"]); C = DEBUG_POSE["C"]
                arq_poses[cid].write(f"{quadro},{t:.4f},{C[0]:.4f},{C[1]:.4f},{C[2]:.4f},{qd[0]:.6f},{qd[1]:.6f},{qd[2]:.6f},{qd[3]:.6f},0,0" + chr(10))
                continue
            if cid == "visao_geral":
                Rv = olhar_para(cen["visao_geral"]["posicao"], cen["visao_geral"]["alvo"])
                qv = quat_de_matriz(Rv)
                C = cen["visao_geral"]["posicao"]
                arq_poses[cid].write(f"{quadro},{t:.4f},{C[0]:.4f},{C[1]:.4f},{C[2]:.4f},{qv[0]:.6f},{qv[1]:.6f},{qv[2]:.6f},{qv[3]:.6f},0,0\n")
                continue
            pose, yaw, cobre = estado[cid]
            qc = quat_de_matriz(pose.R_wc)
            arq_poses[cid].write(f"{quadro},{t:.4f},{pose.C[0]:.4f},{pose.C[1]:.4f},{pose.C[2]:.4f},"
                                 f"{qc[0]:.6f},{qc[1]:.6f},{qc[2]:.6f},{qc[3]:.6f},{yaw:.3f},{int(cobre)}\n")
            bb = a_bb.get_data()
            dados = bb.get("data") if isinstance(bb, dict) else None
            info = bb.get("info", {}) if isinstance(bb, dict) else {}
            caminhos = info.get("primPaths", [])
            if dados is not None:
                for j in range(len(dados)):
                    pid = pid_de_caminho(caminhos[j]) if j < len(caminhos) else None
                    if pid is None:
                        continue
                    r = dados[j]
                    arq_cx[cid].write(f"{quadro},{pid},{float(r['x_min']):.1f},{float(r['y_min']):.1f},"
                                      f"{float(r['x_max']):.1f},{float(r['y_max']):.1f},{float(r['occlusionRatio']):.3f}\n")
        rps_ativos(False)
        if args.depurar and quadro % 10 == 0:
            depurar_pessoas()
        quadro += 1
        if quadro % 25 == 0:
            el = time.time() - t_ini_laco
            log(f"quadro {quadro} t={t:.1f}s; {el / quadro:.2f} s/quadro (render {t_render / quadro:.2f}); "
                f"estimativa total {el / quadro * (n_passos // passo_cap + 1) / 60:.1f} min")
        if args.max_quadros and quadro >= args.max_quadros:
            break
    world.step(render=True)

GRAVADOR.shutdown(wait=True)
for f in list(arq_poses.values()) + list(arq_cx.values()) + list(arq_par.values()) + [arq_gab]:
    f.close()
try:
    rep.orchestrator.wait_until_complete()
except Exception:  # noqa: BLE001
    pass
el = time.time() - t_ini_laco
meta = {
    "simulador": "isaac-sim 5.1.0", "renderer": args.renderer, "rt_subframes": args.rt_subframes,
    "cenario": cen["nome"], "fps": args.fps, "dt": args.dt, "t_inicio": args.t_inicio, "duracao_s": duracao,
    "origem_enu": cen["origem_enu"], "modo_pessoas_pedido": args.modo_pessoas, "modo_pessoas": modo_efetivo,
    "materiais": args.materiais, "ativos": ATIVOS,
    "cameras": {cid: {"tipo": cid if cid in ("visao_geral", "debug") else "drone", "intrinsecos": KS[cid][0].para_dict()} for cid in CAMS},
    "pessoas": list(PERS), "quadros": quadro, "tempo_laco_s": el, "s_por_quadro": el / max(quadro, 1),
    "s_render_por_quadro": t_render / max(quadro, 1), "avisos": AVISOS, "args": vars(args),
    "fonte_gabarito": FONTE_GAB,
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
(OUT / "log_sim.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
log(f"FIM: {quadro} quadros em {el / 60:.1f} min ({el / max(quadro, 1):.2f} s/quadro); saida {OUT}")
app.close()
