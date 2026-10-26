"""Close-up "cinegrafista" no Isaac Sim 5.1 (headless): camera baixa e obliqua que acompanha um grupo de pessoas
animadas (omni.anim.people sobre a navmesh, como no voo run01) enquanto um quadricoptero passa em voo baixo
(~8,5 m). Mostra o realismo da simulacao: personagens animados, RTX em tempo real com sombras, ceu HDRI e
materiais fotograficos no chao. Opcionalmente grava a camera do proprio drone (gimbal) com caixas de gabarito.

Mesmo cenario `reduzido` (terreno/edificios/arvores de cena.json, mesmos comandos de pessoas), mas so com as
pessoas que passam a menos de 60 m do grupo na janela (plano de scripts/planejar_closeup.py).

Rodar (no Spark, via isaac/rodar_isaac.sh, que impoe --memory 24g e o vigia de memoria):
  SCRIPT=/sim/isaac/closeup_isaac.py ATIVOS_LOCAIS=~/cerise2026/drones/sim/isaac_ativos SEM_REDE=1 \\
    ACCEPT_EULA=Y PRIVACY_CONSENT=N bash isaac/rodar_isaac.sh closeup01 --plano /sim/closeup/plano_closeup.json \\
    --hdri /sim/closeup/ceu/kloofendal_48d_partly_cloudy_4k_sem_sol.hdr --heroi 5.0,11.0
Saida: cinegrafista/rgb/*.jpg, gimbal/rgb/*.jpg, gimbal/caixas_gt.csv, heroi/*.png, quadros.csv,
gabarito_pessoas.csv, meta.json, log_sim.txt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--cenario", required=True)
ap.add_argument("--saida", required=True)
ap.add_argument("--plano", required=True, help="JSON de scripts/planejar_closeup.py")
ap.add_argument("--fps", type=float, default=24.0, help="quadros por segundo de simulacao (passo = 1/fps)")
ap.add_argument("--ativos", default=os.environ.get("ISAAC_ASSETS_ROOT"))
ap.add_argument("--hdri", default=None, help=".hdr latlong da cupula (sem o sol; ver scripts/ceu_sem_sol.py)")
ap.add_argument("--ceu_intensidade", type=float, default=760.0)
ap.add_argument("--ceu_rotacao", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="rotateXYZ da DomeLight (graus)")
ap.add_argument("--sol_elevacao", type=float, default=48.0, help="48 graus = elevacao do sol no HDRI kloofendal")
ap.add_argument("--sol_azimute", type=float, default=120.0, help="graus a partir do leste, anti-horario")
ap.add_argument("--sol_intensidade", type=float, default=3500.0)
ap.add_argument("--materiais", choices=["biblioteca", "procedural"], default="biblioteca")
ap.add_argument("--rt_subframes", type=int, default=8)
ap.add_argument("--rt_subframes_heroi", type=int, default=32)
ap.add_argument("--heroi", default="", help="instantes (s) dos quadros de alta qualidade (PNG), ex.: 5.0,11.0")
ap.add_argument("--anti_aliasing", type=int, default=4, help="0 nenhum, 1 TAA, 2 FXAA, 3 DLSS, 4 DLAA")
ap.add_argument("--largura", type=int, default=1920, help="largura da camera do cinegrafista (16:9)")
ap.add_argument("--sem_gimbal", action="store_true")
ap.add_argument("--instantes", default=None, help="teste: grava so estes instantes (s), ex.: 3,6,10,12")
ap.add_argument("--aquecimento_s", type=float, default=3.0, help="igual ao run01: pessoas andam 3 s antes de t = 0")
ap.add_argument("--iso", type=float, default=None, help="sobrescreve /rtx/post/tonemap/filmIso (exposicao)")
ap.add_argument("--qualidade_jpg", type=int, default=95)
args, _desconhecidos = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": 1280, "height": 720,
                     "anti_aliasing": args.anti_aliasing, "multi_gpu": False})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

SIM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM_DIR))
from cerise_sim.camera import Intrinsecos, olhar_para, quat_de_matriz, r_usd_de_cv  # noqa: E402
from cerise_sim.closeup import PlanoCloseup, inclinacao_graus, interp_tabela  # noqa: E402

LOG = []
T_INI_SCRIPT = time.time()


def log(msg):
    s = f"[closeup {time.strftime('%H:%M:%S')}] {msg}"
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

S = carb.settings.get_settings()
S.set("/omni/replicator/captureOnPlay", False)
S.set("/app/runLoops/main/rateLimitEnabled", False)
# qualidade do RTX em tempo real: AO, GI indireta, reflexos, translucidez (disco das helices), cargas sincronas
for k, v in [("/rtx/ambientOcclusion/enabled", True), ("/rtx/indirectDiffuse/enabled", True),
             ("/rtx/reflections/enabled", True), ("/rtx/translucency/enabled", True),
             ("/rtx/raytracing/fractionalCutoutOpacity", True), ("/rtx/shadows/enabled", True),
             ("/rtx/materialDb/syncLoads", True), ("/rtx/hydra/materialSyncLoads", True),
             ("/omni/replicator/asyncRendering", False), ("/rtx/post/dlss/execMode", 2)]:
    try:
        S.set(k, v)
    except Exception as e:  # noqa: BLE001
        log(f"AVISO setting {k}: {e!r}")
if args.iso:
    S.set("/rtx/post/tonemap/filmIso", float(args.iso))
CONFIG_RTX = {k: S.get(k) for k in ["/rtx/rendermode", "/rtx/post/aa/op", "/rtx/post/dlss/execMode", "/rtx/post/tonemap/op",
                                    "/rtx/post/tonemap/filmIso", "/rtx/post/tonemap/cameraShutter", "/rtx/post/tonemap/fNumber",
                                    "/rtx/ambientOcclusion/enabled", "/rtx/indirectDiffuse/enabled", "/rtx/reflections/enabled",
                                    "/rtx/raytracing/fractionalCutoutOpacity", "/rtx/post/histogram/enabled"]}
log(f"RTX: {CONFIG_RTX}")

# ---------------------------------------------------------------------------------------------- plano/cenario
CEN = Path(args.cenario)
cen = json.loads((CEN / "cenario.json").read_text(encoding="utf-8"))
cena = json.loads((CEN / "cena.json").read_text(encoding="utf-8"))
plano_todos = json.loads((CEN / "pessoas_plano.json").read_text(encoding="utf-8"))["pessoas"]
PL = json.loads(Path(args.plano).read_text(encoding="utf-8"))
PC = PlanoCloseup(PL)
IDX_PLANO = {p["id"]: k for k, p in enumerate(plano_todos)}  # mesmo personagem do run01 (indice no plano completo)
plano_p = [p for p in plano_todos if p["id"] in set(PL["pessoas"])]
GRUPO = PL["grupo"]
OUT = Path(args.saida)
OUT.mkdir(parents=True, exist_ok=True)
DT = 1.0 / args.fps


def raiz_ativos():
    if args.ativos:
        return args.ativos.rstrip("/")
    from isaacsim.storage.native import get_assets_root_path
    return get_assets_root_path().rstrip("/")


ATIVOS = raiz_ativos()
log(f"ativos {ATIVOS!r}; {len(plano_p)} pessoas (grupo {GRUPO}); janela {PL['t_inicio']}-{PL['t_fim']} s a {args.fps} fps")

ctx = omni.usd.get_context()
ctx.new_stage()
app.update()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
for p in ["/World", "/World/Terreno", "/World/Edificios", "/World/Arvores", "/World/Cameras", "/World/Looks"]:
    UsdGeom.Xform.Define(stage, p)
stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))


def omnipbr(nome, cor=(0.5, 0.5, 0.5), rug=0.8, metal=0.0, difusa=None, orm=None, normal=None, opacidade=None,
            emissao=None, tinta=None, brilho=None, projecao_mundo_m=None, especular=None):
    caminho = f"/World/Looks/{nome}"
    omni.kit.commands.execute("CreateMdlMaterialPrimCommand", mtl_url="OmniPBR.mdl", mtl_name="OmniPBR", mtl_path=caminho)
    sh = UsdShade.Shader(stage.GetPrimAtPath(caminho + "/Shader"))

    def inp(n, tipo, v):
        sh.CreateInput(n, tipo).Set(v)
    inp("diffuse_color_constant", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*cor))
    inp("reflection_roughness_constant", Sdf.ValueTypeNames.Float, float(rug))
    inp("metallic_constant", Sdf.ValueTypeNames.Float, float(metal))
    if difusa:
        inp("diffuse_texture", Sdf.ValueTypeNames.Asset, str(difusa))
    if orm:
        inp("enable_ORM_texture", Sdf.ValueTypeNames.Bool, True)
        inp("ORM_texture", Sdf.ValueTypeNames.Asset, str(orm))
    if normal:
        inp("normalmap_texture", Sdf.ValueTypeNames.Asset, str(normal))
        inp("bump_factor", Sdf.ValueTypeNames.Float, 1.0)
    if opacidade is not None:
        inp("enable_opacity", Sdf.ValueTypeNames.Bool, True)
        inp("opacity_constant", Sdf.ValueTypeNames.Float, float(opacidade))
        inp("opacity_threshold", Sdf.ValueTypeNames.Float, 0.0)
    if emissao is not None:
        inp("enable_emission", Sdf.ValueTypeNames.Bool, True)
        inp("emissive_color", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*emissao[0]))
        inp("emissive_intensity", Sdf.ValueTypeNames.Float, float(emissao[1]))
    if tinta is not None:
        inp("diffuse_tint", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*tinta))
    if brilho is not None:
        inp("albedo_brightness", Sdf.ValueTypeNames.Float, float(brilho))
    if especular is not None:
        inp("specular_level", Sdf.ValueTypeNames.Float, float(especular))
    if projecao_mundo_m:  # textura projetada em coordenadas do mundo (primitivas sem UV, ex.: copas)
        inp("project_uvw", Sdf.ValueTypeNames.Bool, True)
        inp("world_or_object", Sdf.ValueTypeNames.Bool, True)
        inp("texture_scale", Sdf.ValueTypeNames.Float2, Gf.Vec2f(projecao_mundo_m, projecao_mundo_m))
    return UsdShade.Material(stage.GetPrimAtPath(caminho))


# materiais do chao: texturas fotograficas da biblioteca NVIDIA (Base/Natural, Base/Masonry) e ladrilho em metros
BIB = {  # nome no cenario -> (pasta, prefixo, metros por ladrilho, tinta)
    "grama": ("Natural/Grass_Cut", "Grass_Cut", 3.0, (0.80, 0.84, 0.72)),
    "grama_seca": ("Natural/Grass_Countryside", "Grass_Countryside", 4.0, (0.85, 0.85, 0.75)),
    "terra": ("Natural/Dirt", "Dirt", 2.5, None),
    "concreto": ("Masonry/Concrete_Smooth", "Concrete_Smooth", 2.5, (0.80, 0.79, 0.76)),
    "asfalto": ("Natural/Asphalt", "Asphalt", 4.0, None),
    "parede": ("Masonry/Concrete_Smooth", "Concrete_Smooth", 3.0, (0.92, 0.84, 0.70)),
}


def textura_bib(nome_bib):
    pasta, pre, lad, tinta = BIB[nome_bib]
    base = f"{ATIVOS}/NVIDIA/Materials/Base/{pasta}/{pre}"
    nrm = f"{base}_Normal.png" if os.path.exists(f"{base}_Normal.png") else f"{base}_N.png"
    return (base if os.path.exists(f"{base}_BaseColor.png") else None), nrm, lad, tinta


MATS, LADRILHO = {}, {}
for nome, info in cen.get("texturas", {}).items():
    mpl = info.get("metros_por_ladrilho", 8.0)
    if args.materiais == "biblioteca" and nome in BIB:
        base, nrm, lad, tinta = textura_bib(nome)
        if base:
            MATS[nome] = omnipbr(nome, cor=(1, 1, 1), rug=1.0, difusa=f"{base}_BaseColor.png", orm=f"{base}_ORM.png", normal=nrm,
                                 tinta=tinta)
            LADRILHO[nome] = mpl / lad  # fator aplicado as UVs do cenario (que estao em ladrilhos de mpl metros)
            continue
        log(f"AVISO textura da biblioteca ausente para {nome}; usando a procedural")
    MATS[nome] = omnipbr(nome, rug=info.get("rugosidade", 0.8), difusa=(CEN / info["arquivo"]).resolve())
    LADRILHO[nome] = 1.0
MATS["tronco"] = omnipbr("tronco", cor=(0.27, 0.20, 0.13), rug=0.95)
_bc, _nc, _l, _t = textura_bib("grama_seca") if args.materiais == "biblioteca" else (None, None, None, None)
if _bc:  # copa com textura de vegetacao projetada no mundo (as esferas nao tem UV), tons escuros diferentes
    MATS["copa"] = omnipbr("copa", cor=(1, 1, 1), rug=0.9, difusa=f"{_bc}_BaseColor.png", normal=_nc, tinta=(0.42, 0.55, 0.30),
                           projecao_mundo_m=1.6)
    MATS["copa2"] = omnipbr("copa2", cor=(1, 1, 1), rug=0.9, difusa=f"{_bc}_BaseColor.png", normal=_nc, tinta=(0.50, 0.62, 0.34),
                            projecao_mundo_m=2.1)
else:
    MATS["copa"] = omnipbr("copa", cor=(0.10, 0.21, 0.07), rug=0.9)
    MATS["copa2"] = omnipbr("copa2", cor=(0.14, 0.26, 0.09), rug=0.9)
MATS["vidro"] = omnipbr("vidro", cor=(0.05, 0.07, 0.09), rug=0.06, metal=0.0, especular=0.9)
MATS["caixilho"] = omnipbr("caixilho", cor=(0.55, 0.56, 0.57), rug=0.4, metal=0.6)


def malha(caminho, V, F, uv=None, material=None):
    m = UsdGeom.Mesh.Define(stage, caminho)
    m.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in V])
    m.CreateFaceVertexCountsAttr([len(f) for f in F])
    m.CreateFaceVertexIndicesAttr([int(i) for f in F for i in f])
    m.CreateSubdivisionSchemeAttr("none")
    if uv is not None:
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set([Gf.Vec2f(float(a), float(b)) for a, b in uv])
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    return m


t0 = time.time()
for mm in cena["malhas"]:
    if mm["vertices"]:
        f = LADRILHO.get(mm["material"], 1.0)
        malha(f"/World/Terreno/{mm['nome']}", mm["vertices"], mm["faces"], np.asarray(mm["uv"]) * f, MATS.get(mm["material"]))
mpl_parede = cen["texturas"].get("parede", {}).get("metros_por_ladrilho", 6.0)
mpl_telhado = cen["texturas"].get("telhado", {}).get("metros_por_ladrilho", 8.0)
for i, e in enumerate(cena["edificios"]):
    mm = e["malha"]
    V = np.array(mm["vertices"]); F = mm["faces"]; UV = np.array(mm["uv_metros"])
    npf = mm["faces_parede"]
    idx_par = sorted({k for f in F[:npf] for k in f}); idx_tel = sorted({k for f in F[npf:] for k in f})
    for nome, idx, faces, mat, esc in [("paredes", idx_par, F[:npf], "parede", mpl_parede), ("telhado", idx_tel, F[npf:], "telhado", mpl_telhado)]:
        mapa = {k: j for j, k in enumerate(idx)}
        malha(f"/World/Edificios/E{i:02d}_{nome}", V[idx], [[mapa[k] for k in f] for f in faces],
              UV[idx] / esc * LADRILHO.get(mat, 1.0), MATS.get(mat))
    # faixas de janelas por andar (vidro escuro reflexivo) e montantes, a 4-7 cm da fachada
    peg = np.asarray(e["pegada"], float)
    area2 = float(np.sum(peg[:, 0] * np.roll(peg[:, 1], -1) - np.roll(peg[:, 0], -1) * peg[:, 1]))
    hf = e["altura_m"] / max(e.get("andares", 1), 1)
    Vv, Fv, Vm, Fm = [], [], [], []
    for a_, b_ in zip(peg, np.roll(peg, -1, axis=0)):
        L = float(np.linalg.norm(b_ - a_))
        if L < 4.0:
            continue
        u = (b_ - a_) / L
        n = np.array([u[1], -u[0]]) if area2 > 0 else np.array([-u[1], u[0]])  # normal para fora
        for j in range(max(e.get("andares", 1), 1)):
            z0 = j * hf + 0.9
            z1 = z0 + min(1.7, hf - 1.3)
            if z1 <= z0:
                continue
            p0, p1 = a_ + u * 0.8 + n * 0.04, b_ - u * 0.8 + n * 0.04
            o = len(Vv)
            Vv += [[p0[0], p0[1], z0], [p1[0], p1[1], z0], [p1[0], p1[1], z1], [p0[0], p0[1], z1]]
            Fv.append([o, o + 1, o + 2, o + 3])
            for s_ in np.arange(0.8, L - 0.8 + 1e-6, 1.6):
                q0, q1 = a_ + u * (s_ - 0.06) + n * 0.07, a_ + u * (s_ + 0.06) + n * 0.07
                o = len(Vm)
                Vm += [[q0[0], q0[1], z0 - 0.08], [q1[0], q1[1], z0 - 0.08], [q1[0], q1[1], z1 + 0.08], [q0[0], q0[1], z1 + 0.08]]
                Fm.append([o, o + 1, o + 2, o + 3])
    if Fv:
        malha(f"/World/Edificios/E{i:02d}_janelas", Vv, Fv, None, MATS["vidro"]).CreateDoubleSidedAttr(True)
        malha(f"/World/Edificios/E{i:02d}_montantes", Vm, Fm, None, MATS["caixilho"]).CreateDoubleSidedAttr(True)
# arvores: tronco + copa em cachos de elipsoides (mais organica que a esfera unica do run01)
rnd = random.Random(7)
for i, a in enumerate(cena["arvores"]):
    base = f"/World/Arvores/A{i:03d}"
    UsdGeom.Xform.Define(stage, base)
    h_tronco = a["altura_m"] * 0.45
    rc = a["raio_copa_m"]
    tr = UsdGeom.Cylinder.Define(stage, base + "/tronco")
    tr.CreateRadiusAttr(0.16 + 0.03 * rc); tr.CreateHeightAttr(h_tronco + 0.6 * rc); tr.CreateAxisAttr("Z")
    UsdGeom.XformCommonAPI(tr).SetTranslate(Gf.Vec3d(a["x"], a["y"], (h_tronco + 0.6 * rc) / 2))
    UsdShade.MaterialBindingAPI.Apply(tr.GetPrim()).Bind(MATS["tronco"])
    zc = h_tronco + 0.6 * rc
    hz = 0.75 * rc + 0.25 * (a["altura_m"] - h_tronco)
    for j in range(7):
        ang = rnd.uniform(0, 2 * math.pi)
        rr = rc * (0.0 if j == 0 else rnd.uniform(0.35, 0.6))
        esc = rc * (0.75 if j == 0 else rnd.uniform(0.45, 0.62))
        cp = UsdGeom.Sphere.Define(stage, f"{base}/copa{j}")
        cp.CreateRadiusAttr(1.0)
        api = UsdGeom.XformCommonAPI(cp)
        api.SetTranslate(Gf.Vec3d(a["x"] + rr * math.cos(ang), a["y"] + rr * math.sin(ang), zc + rnd.uniform(-0.15, 0.3) * hz))
        api.SetScale(Gf.Vec3f(esc, esc * rnd.uniform(0.85, 1.1), hz * (0.8 if j == 0 else rnd.uniform(0.5, 0.7))))
        UsdShade.MaterialBindingAPI.Apply(cp.GetPrim()).Bind(MATS["copa" if j % 2 == 0 else "copa2"])
log(f"terreno: {len(cena['malhas'])} malhas, {len(cena['edificios'])} edificios, {len(cena['arvores'])} arvores "
    f"({time.time() - t0:.1f} s); materiais {args.materiais}; fatores de UV {LADRILHO}")

# luz: sol explicito (DistantLight, angulo do disco solar real) + cupula HDRI sem o sol
sol = UsdLux.DistantLight.Define(stage, "/World/Sol")
sol.CreateIntensityAttr(args.sol_intensidade); sol.CreateAngleAttr(0.53); sol.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
el, az = math.radians(args.sol_elevacao), math.radians(args.sol_azimute)
d_sol = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
q = quat_de_matriz(r_usd_de_cv(olhar_para(d_sol * 1000.0, np.zeros(3))))
UsdGeom.Xformable(sol).AddOrientOp().Set(Gf.Quatf(*map(float, q)))
ceu = UsdLux.DomeLight.Define(stage, "/World/Ceu")
ceu.CreateIntensityAttr(args.ceu_intensidade)
if args.hdri:
    ceu.CreateTextureFileAttr().Set(args.hdri)
    ceu.CreateTextureFormatAttr().Set("latlong")
    UsdGeom.XformCommonAPI(ceu).SetRotate(Gf.Vec3f(*args.ceu_rotacao))
else:
    ceu.CreateColorAttr(Gf.Vec3f(0.78, 0.86, 1.0))


# ---------------------------------------------------------------------------------------------- drone detalhado
def drone_detalhado(raiz, env=0.9):
    """Quadricoptero de inspecao (~0,9 m motor a motor, helices de 21"), materiais PBR, sombra ligada.
    +X local = frente. Devolve os Xforms dos rotores (para girar) e o sentido de cada um."""
    M = {"corpo": omnipbr("dr_corpo", cor=(0.06, 0.065, 0.07), rug=0.45),
         "tampa": omnipbr("dr_tampa", cor=(0.55, 0.57, 0.6), rug=0.35, metal=0.1),
         "carbono": omnipbr("dr_carbono", cor=(0.025, 0.025, 0.03), rug=0.3),
         "motor": omnipbr("dr_motor", cor=(0.35, 0.36, 0.38), rug=0.25, metal=0.9),
         "helice": omnipbr("dr_helice", cor=(0.03, 0.03, 0.03), rug=0.5),
         "disco": omnipbr("dr_disco", cor=(0.05, 0.05, 0.05), rug=0.6, opacidade=0.10),
         "trem": omnipbr("dr_trem", cor=(0.75, 0.76, 0.78), rug=0.4, metal=0.3),
         "lente": omnipbr("dr_lente", cor=(0.02, 0.02, 0.04), rug=0.05, metal=0.2),
         "led_r": omnipbr("dr_led_r", cor=(1, 0.1, 0.1), emissao=((1.0, 0.08, 0.05), 6000.0)),
         "led_g": omnipbr("dr_led_g", cor=(0.1, 1, 0.1), emissao=((0.1, 1.0, 0.2), 6000.0))}
    UsdGeom.Xform.Define(stage, raiz)

    def prim(tipo, nome, mat, tr=(0, 0, 0), esc=(1, 1, 1), rot=(0, 0, 0), **kw):
        p = tipo.Define(stage, f"{raiz}/{nome}")
        for k, v in kw.items():
            getattr(p, f"Create{k}Attr")(v)
        api = UsdGeom.XformCommonAPI(p)
        api.SetTranslate(Gf.Vec3d(*tr)); api.SetRotate(Gf.Vec3f(*rot)); api.SetScale(Gf.Vec3f(*esc))
        UsdShade.MaterialBindingAPI.Apply(p.GetPrim()).Bind(M[mat])
        return p
    prim(UsdGeom.Cube, "fuselagem", "corpo", esc=(0.30, 0.19, 0.09), Size=1.0)
    prim(UsdGeom.Sphere, "tampa", "tampa", tr=(0.0, 0, 0.045), esc=(0.17, 0.10, 0.045), Radius=1.0)
    prim(UsdGeom.Cube, "bateria", "corpo", tr=(-0.05, 0, 0.075), esc=(0.16, 0.12, 0.05), Size=1.0)
    prim(UsdGeom.Cylinder, "antena_e", "corpo", tr=(-0.13, 0.07, 0.09), esc=(1, 1, 1), Radius=0.008, Height=0.10, Axis="Z")
    prim(UsdGeom.Cylinder, "antena_d", "corpo", tr=(-0.13, -0.07, 0.09), esc=(1, 1, 1), Radius=0.008, Height=0.10, Axis="Z")
    r_m = env / 2
    rotores = []
    for k, angd in enumerate((45, 135, 225, 315)):
        a = math.radians(angd)
        cx, cy = r_m * math.cos(a), r_m * math.sin(a)
        comp = r_m - 0.10
        mx, my = (0.10 + comp / 2) * math.cos(a), (0.10 + comp / 2) * math.sin(a)
        prim(UsdGeom.Capsule, f"braco{k}", "carbono", tr=(mx, my, 0.01), rot=(0, 90, angd), Radius=0.017, Height=comp, Axis="Z")
        prim(UsdGeom.Cylinder, f"motor{k}", "motor", tr=(cx, cy, 0.035), Radius=0.034, Height=0.05, Axis="Z")
        prim(UsdGeom.Sphere, f"led{k}", "led_r" if angd in (45, 315) else "led_g", tr=(cx + 0.03 * math.cos(a), cy + 0.03 * math.sin(a), -0.005),
             Radius=0.012)
        rx = f"{raiz}/rotor{k}"
        xr = UsdGeom.Xform.Define(stage, rx)
        xr.AddTranslateOp().Set(Gf.Vec3d(cx, cy, 0.068))
        rz = xr.AddRotateZOp()
        for b in range(2):
            pa = UsdGeom.Cube.Define(stage, f"{rx}/pa{b}")
            pa.CreateSizeAttr(1.0)
            api = UsdGeom.XformCommonAPI(pa)
            api.SetRotate(Gf.Vec3f(8.0 if b == 0 else -8.0, 0, 180.0 * b))
            api.SetTranslate(Gf.Vec3d(0.0675 * (1 if b == 0 else -1), 0, 0))
            api.SetScale(Gf.Vec3f(0.135, 0.032, 0.006))
            UsdShade.MaterialBindingAPI.Apply(pa.GetPrim()).Bind(M["helice"])
            pb = UsdGeom.Cube.Define(stage, f"{rx}/ponta{b}")
            pb.CreateSizeAttr(1.0)
            api = UsdGeom.XformCommonAPI(pb)
            api.SetTranslate(Gf.Vec3d(0.20 * (1 if b == 0 else -1), 0, 0))
            api.SetScale(Gf.Vec3f(0.13, 0.026, 0.005))
            UsdShade.MaterialBindingAPI.Apply(pb.GetPrim()).Bind(M["helice"])
        cubo = UsdGeom.Cylinder.Define(stage, f"{rx}/cubo")
        cubo.CreateRadiusAttr(0.018); cubo.CreateHeightAttr(0.02); cubo.CreateAxisAttr("Z")
        UsdShade.MaterialBindingAPI.Apply(cubo.GetPrim()).Bind(M["motor"])
        disco = UsdGeom.Cylinder.Define(stage, f"{rx}/disco")
        disco.CreateRadiusAttr(0.265); disco.CreateHeightAttr(0.002); disco.CreateAxisAttr("Z")
        UsdShade.MaterialBindingAPI.Apply(disco.GetPrim()).Bind(M["disco"])
        UsdGeom.PrimvarsAPI(disco).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        rotores.append((rz, 1 if k % 2 == 0 else -1))
    # trem de pouso (dois esquis) e gimbal com camera sob a frente
    for s in (1, -1):
        lado = "e" if s > 0 else "d"
        prim(UsdGeom.Capsule, f"esqui_{lado}", "trem", tr=(0, 0.17 * s, -0.25), rot=(0, 90, 0), Radius=0.012, Height=0.40, Axis="Z")
        for xx in (0.09, -0.09):
            prim(UsdGeom.Cylinder, f"perna_{lado}{0 if xx > 0 else 1}", "trem", tr=(xx, 0.13 * s, -0.15), rot=(22 * s, 0, 0),
                 Radius=0.011, Height=0.22, Axis="Z")
    prim(UsdGeom.Cube, "gimbal_suporte", "corpo", tr=(0.11, 0, -0.07), esc=(0.03, 0.05, 0.05), Size=1.0)
    prim(UsdGeom.Sphere, "gimbal", "corpo", tr=(0.12, 0, -0.11), Radius=0.045)
    prim(UsdGeom.Cylinder, "lente", "lente", tr=(0.16, 0, -0.115), rot=(0, 90, 0), Radius=0.02, Height=0.02, Axis="Z")
    return rotores


ROTORES = drone_detalhado("/World/DroneCloseup", PL["drone"].get("envergadura_m", 0.9))
XF_DRONE = UsdGeom.Xformable(stage.GetPrimAtPath("/World/DroneCloseup"))
XF_DRONE.ClearXformOpOrder()
OP_DR_T = XF_DRONE.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
OP_DR_R = XF_DRONE.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
OP_DR_T.Set(Gf.Vec3d(*map(float, PC.drone(PL["t_inicio"]))))  # fora do volume da navmesh desde o inicio


def por_drone(pos, rumo_graus, incl_graus, t):
    """rumo a partir do leste (anti-horario); incl > 0 = nariz para baixo (multirrotor acelerando para a frente)."""
    cy, sy = math.cos(math.radians(rumo_graus) / 2), math.sin(math.radians(rumo_graus) / 2)
    cp, sp = math.cos(math.radians(incl_graus) / 2), math.sin(math.radians(incl_graus) / 2)
    qz = Gf.Quatd(cy, 0, 0, sy)
    qy = Gf.Quatd(cp, 0, sp, 0)  # rotacao em torno de +Y local: +X (frente) desce
    OP_DR_T.Set(Gf.Vec3d(*map(float, pos)))
    OP_DR_R.Set(qz * qy)
    hz = PL["drone"].get("rotacao_helices_hz", 7.0)
    for k, (op, s) in enumerate(ROTORES):
        op.Set(float((s * 360.0 * hz * t + 37.0 * k) % 360.0))


# ---------------------------------------------------------------------------------------------- cameras
def camera_usd(caminho, K, perto=0.05):
    cam = UsdGeom.Camera.Define(stage, caminho)
    u = K.usd()
    cam.CreateFocalLengthAttr(u["focalLength"]); cam.CreateHorizontalApertureAttr(u["horizontalAperture"])
    cam.CreateVerticalApertureAttr(u["verticalAperture"]); cam.CreateClippingRangeAttr(Gf.Vec2f(perto, 5000.0))
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    return cam, xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble), xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)


ALT_CINE = int(round(args.largura * 9 / 16 / 2)) * 2
K_CINE = Intrinsecos.de_hfov(args.largura, ALT_CINE, PL["camera"]["hfov"])
CAM_CINE, OP_C_T, OP_C_R = camera_usd("/World/Cameras/cinegrafista", K_CINE, 0.1)
G = PL["gimbal"]
K_GIM = Intrinsecos.de_hfov(G["largura"] * args.largura // 1920, G["altura"] * args.largura // 1920, G["hfov"])
CAM_GIM, OP_G_T, OP_G_R = (None, None, None) if args.sem_gimbal else camera_usd("/World/Cameras/gimbal", K_GIM, 0.3)


def por_camera(op_t, op_r, C, alvo):
    R = olhar_para(C, alvo)
    q = quat_de_matriz(r_usd_de_cv(R))
    op_t.Set(Gf.Vec3d(*map(float, C)))
    op_r.Set(Gf.Quatd(*map(float, q)))
    return R


# ---------------------------------------------------------------------------------------------- pessoas (como no run01)
PERS, SKEL, AVISOS = {}, {}, []


def listar_personagens():
    import omni.client
    base = f"{ATIVOS}/Isaac/People/Characters"
    res, itens = omni.client.list(base + "/")
    pastas = sorted(i.relative_path.rstrip("/") for i in itens if (i.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN)
                    and not i.relative_path.startswith(".") and not i.relative_path.lower().startswith("biped")
                    and not i.relative_path.startswith("DH_"))
    usds = []
    for p in pastas:
        _r, fs = omni.client.list(f"{base}/{p}/")
        u = sorted(f.relative_path for f in fs if f.relative_path.endswith((".usd", ".usda")) and not f.relative_path.startswith("."))
        if u:
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
    UsdGeom.Xform.Define(stage, "/World/Characters")
    usds = listar_personagens()
    for p in plano_p:
        caminho = f"/World/Characters/{p['id']}"
        xf = UsdGeom.Xform.Define(stage, caminho)
        xf.GetPrim().GetReferences().AddReference(usds[IDX_PLANO[p["id"]] % len(usds)])
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
    import NavSchema
    for caminho in PERS.values():
        omni.kit.commands.execute("ApplyNavMeshAPICommand", prim_path=caminho, api=NavSchema.NavMeshExcludeAPI)
    omni.kit.commands.execute("ApplyNavMeshAPICommand", prim_path="/World/DroneCloseup", api=NavSchema.NavMeshExcludeAPI)
    biped = UsdGeom.Xform.Define(stage, "/World/Characters/Biped_Setup")
    biped.GetPrim().GetReferences().AddReference(f"{ATIVOS}/Isaac/People/Characters/Biped_Setup.usd")
    UsdGeom.Imageable(biped).MakeInvisible()
    app.update()
    from omni.anim.people.scripts.custom_command.populate_anim_graph import populate_anim_graph
    populate_anim_graph()
    grafo = next((prim.GetPath() for prim in Usd.PrimRange(biped.GetPrim()) if prim.GetTypeName() == "AnimationGraph"), None)
    if grafo is None:
        raise RuntimeError("AnimationGraph nao encontrado no Biped_Setup")
    caminhos = [Sdf.Path(p) for p in SKEL.values()]
    omni.kit.commands.execute("RemoveAnimationGraphAPICommand", paths=caminhos)
    omni.kit.commands.execute("ApplyAnimationGraphAPICommand", paths=caminhos, animation_graph_path=grafo)
    mgr = omni.kit.app.get_app().get_extension_manager()
    script = mgr.get_extension_path_by_module("omni.anim.people") + "/omni/anim/people/scripts/character_behavior.py"
    omni.kit.commands.execute("RemoveScriptingAPICommand", paths=caminhos)
    omni.kit.commands.execute("ApplyScriptingAPICommand", paths=caminhos)
    for pth in SKEL.values():
        stage.GetPrimAtPath(pth).GetAttribute("omni:scripting:scripts").Set([script])
    cmd = OUT / "pessoas_comandos_usados.txt"
    ids = set(PERS)
    linhas = [ln for ln in (CEN / "pessoas_comandos.txt").read_text(encoding="utf-8").splitlines() if ln.split(" ")[0] in ids]
    cmd.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    S.set("/exts/omni.anim.people/command_settings/command_file_path", str(cmd))
    S.set("/persistent/exts/omni.anim.people/character_prim_path", "/World/Characters")
    S.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
    S.set("/exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled", True)
    log(f"pessoas criadas: {len(PERS)} (SkelRoots: {len(SKEL)})")


def assar_navmesh():
    """mesma navmesh do run01 (volume do gramado + 20 m), para os caminhos das pessoas serem os mesmos."""
    import omni.anim.navigation.core as nav
    omni.kit.commands.execute("CreateNavMeshVolumeCommand", parent_prim_path=Sdf.Path("/World"), volume_type=0)
    app.update()
    vol = next((p for p in stage.Traverse() if p.GetTypeName() == "NavMeshVolume"), None)
    xs = [v[0] for mm in cena["malhas"] if mm["nome"] == "gramado" for v in mm["vertices"]]
    ys = [v[1] for mm in cena["malhas"] if mm["nome"] == "gramado" for v in mm["vertices"]]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    sx, sy = (max(xs) - min(xs)) / 2 + 20, (max(ys) - min(ys)) / 2 + 20
    UsdGeom.Boundable(vol).GetExtentAttr().Set([Gf.Vec3f(-sx, -sy, -3.0), Gf.Vec3f(sx, sy, 3.0)])
    xf = UsdGeom.Xformable(vol)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(cx, cy, 1.0))
    app.update()
    t = time.time()
    inav = nav.acquire_interface()
    inav.start_navmesh_baking_and_wait()
    ok = inav.get_navmesh() is not None
    log(f"navmesh assada: {ok} ({time.time() - t:.1f} s), ~{2 * sx:.0f} x {2 * sy:.0f} m")
    if not ok:
        raise RuntimeError("navmesh nao assada")


criar_pessoas()
assar_navmesh()
_AG = {}


def personagem(pid):
    if pid not in _AG:
        try:
            import omni.anim.graph.core as ag
            _AG[pid] = ag.get_character(SKEL[pid])
        except Exception:  # noqa: BLE001
            _AG[pid] = None
    return _AG[pid]


def posicoes_reais():
    out = {}
    for pid in PERS:
        c = personagem(pid)
        if c is None:
            continue
        pos = carb.Float3(0, 0, 0); rot = carb.Float4(0, 0, 0, 0)
        c.get_world_transform(pos, rot)
        if abs(pos.x) + abs(pos.y) > 1e-6:
            out[pid] = (pos.x, pos.y, pos.z, math.degrees(2 * math.atan2(rot.z, rot.w)))
    return out


# ---------------------------------------------------------------------------------------------- replicator
RPS = []
rp_c = rep.create.render_product(str(CAM_CINE.GetPath()), (K_CINE.largura, K_CINE.altura), name="rp_cinegrafista")
RPS.append(rp_c)
A_RGB_C = rep.AnnotatorRegistry.get_annotator("rgb"); A_RGB_C.attach(rp_c)
(OUT / "cinegrafista" / "rgb").mkdir(parents=True, exist_ok=True)
(OUT / "heroi").mkdir(parents=True, exist_ok=True)
if CAM_GIM is not None:
    rp_g = rep.create.render_product(str(CAM_GIM.GetPath()), (K_GIM.largura, K_GIM.altura), name="rp_gimbal")
    RPS.append(rp_g)
    A_RGB_G = rep.AnnotatorRegistry.get_annotator("rgb"); A_RGB_G.attach(rp_g)
    A_BB_G = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight", init_params={"semanticTypes": ["class"]}); A_BB_G.attach(rp_g)
    (OUT / "gimbal" / "rgb").mkdir(parents=True, exist_ok=True)


def rps_ativos(on):
    for rp in RPS:
        try:
            rp.hydra_texture.set_updates_enabled(on)
        except Exception:  # noqa: BLE001
            pass


GRAVADOR = ThreadPoolExecutor(max_workers=4)


def salvar(arr, caminho):
    from PIL import Image
    img = Image.fromarray(np.ascontiguousarray(arr[..., :3]))
    if str(caminho).endswith(".png"):
        img.save(caminho, compress_level=4)
    else:
        img.save(caminho, quality=args.qualidade_jpg, subsampling=0)


def pid_de_caminho(c):
    partes = str(c).split("/")
    return partes[partes.index("Characters") + 1] if "Characters" in partes and partes.index("Characters") + 1 < len(partes) else None


# ---------------------------------------------------------------------------------------------- laco
from isaacsim.core.api import World  # noqa: E402

world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
world.reset()
world.play()
rps_ativos(False)
T0, T1 = PL["t_inicio"], PL["t_fim"]
instantes = sorted(float(x) for x in args.instantes.split(",")) if args.instantes else None
herois = sorted(float(x) for x in args.heroi.split(",") if x.strip()) if args.heroi else []
k_heroi = {int(round((h + args.aquecimento_s) / DT)) for h in herois}
k_inst = {int(round((h + args.aquecimento_s) / DT)) for h in instantes} if instantes else None
k_ini = int(round((T0 + args.aquecimento_s) / DT))
k_fim = int(round((T1 + args.aquecimento_s) / DT))
if k_inst:
    k_fim = max(k_inst)
log(f"passo {DT:.4f} s; avanco sem gravar ate t = {T0} s (+{args.aquecimento_s} s de aquecimento): {k_ini} passos; "
    f"gravacao {'nos instantes ' + str(instantes) if instantes else f'{k_ini}..{k_fim}'}; herois {herois}")

cen_pred = PL["centroide_pred"]
desvio = None          # desvio (suavizado) entre o centroide medido do grupo e o previsto pelo run01
alpha = 1 - math.exp(-DT / PL["camera"].get("suavizacao_s", 1.5))
arq_q = open(OUT / "quadros.csv", "w", encoding="utf-8")
arq_q.write("quadro,t,cx,cy,cz,alvo_x,alvo_y,alvo_z,hfov,inclinacao,dx,dy,dz,drone_rumo,drone_incl,drone_v,"
            "gcx,gcy,gcz,centro_x,centro_y,desvio_x,desvio_y,subframes,s_render\n")
arq_gab = open(OUT / "gabarito_pessoas.csv", "w", encoding="utf-8")
arq_gab.write("quadro,t,pessoa_id,x,y,z,yaw\n")
arq_cx = open(OUT / "gimbal" / "caixas_gt.csv", "w", encoding="utf-8") if CAM_GIM is not None else None
if arq_cx:
    arq_cx.write("quadro,pessoa_id,x_min,y_min,x_max,y_max,ocl\n")

t_loop0 = time.time()
t_avanco = None
t_render = 0.0
quadro = 0
for k in range(k_fim + 1):
    t = k * DT - args.aquecimento_s    # mesmo relogio do gabarito do run01 (t = 0 depois de 3 s de aquecimento)
    gravar = (k_inst is not None and k in k_inst) or (k_inst is None and k_ini <= k <= k_fim)
    if k >= k_ini - 2 or gravar:
        reais = posicoes_reais()
        g = [reais[p] for p in GRUPO if p in reais]
        centro_med = np.mean([[x, y] for x, y, _z, _w in g], axis=0) if g else None
        pred = interp_tabela(cen_pred, t)
        if centro_med is not None:
            off = centro_med - pred
            desvio = off if desvio is None else desvio + alpha * (off - desvio)
        centro = pred + (desvio if desvio is not None else 0.0)
    if not gravar:
        world.step(render=True)
        continue
    if t_avanco is None:
        t_avanco = time.time() - t_loop0
        log(f"avanco ate t = {t:.2f} s em {t_avanco:.1f} s; desvio do grupo frente ao run01: {np.round(desvio, 2).tolist()} m")
    off3 = np.array([desvio[0], desvio[1], 0.0]) if desvio is not None else np.zeros(3)
    dpos, rumo, incl, vh = PC.drone_estado(t)
    dpos = dpos + off3
    por_drone(dpos, rumo, incl, t)
    C, alvo = PC.camera(t, centro, dpos)
    hf = PC.hfov(t)
    Kt = Intrinsecos.de_hfov(K_CINE.largura, K_CINE.altura, hf)
    CAM_CINE.GetFocalLengthAttr().Set(Kt.usd()["focalLength"])
    por_camera(OP_C_T, OP_C_R, C, alvo)
    # gimbal: 12 cm a frente e 13 cm abaixo do centro do drone, apontado para o grupo
    cr, sr = math.cos(math.radians(rumo)), math.sin(math.radians(rumo))
    Cg = dpos + np.array([0.12 * cr, 0.12 * sr, -0.13])
    alvo_g = np.array([centro[0], centro[1], 0.9])
    if CAM_GIM is not None:
        por_camera(OP_G_T, OP_G_R, Cg, alvo_g)
    if quadro == 0:
        # aquece texturas/materiais vistos pelas cameras (descartado)
        rps_ativos(True)
        rep.orchestrator.step(rt_subframes=max(args.rt_subframes_heroi, 16), delta_time=0.0, pause_timeline=False)
        rep.orchestrator.step(rt_subframes=max(args.rt_subframes_heroi, 16), delta_time=0.0, pause_timeline=False)
    heroi = k in k_heroi
    sub = args.rt_subframes_heroi if heroi else args.rt_subframes
    rps_ativos(True)
    tr0 = time.time()
    rep.orchestrator.step(rt_subframes=sub, delta_time=0.0, pause_timeline=False)
    dtr = time.time() - tr0
    t_render += dtr
    rgb = A_RGB_C.get_data()
    if rgb is not None and getattr(rgb, "size", 0):
        arr = np.array(rgb)
        GRAVADOR.submit(salvar, arr, OUT / "cinegrafista" / "rgb" / f"{quadro:06d}.jpg")
        if heroi:
            GRAVADOR.submit(salvar, arr, OUT / "heroi" / f"cinegrafista_t{t:05.2f}.png")
    if CAM_GIM is not None:
        rgb = A_RGB_G.get_data()
        if rgb is not None and getattr(rgb, "size", 0):
            arr = np.array(rgb)
            GRAVADOR.submit(salvar, arr, OUT / "gimbal" / "rgb" / f"{quadro:06d}.jpg")
            if heroi:
                GRAVADOR.submit(salvar, arr, OUT / "heroi" / f"gimbal_t{t:05.2f}.png")
        bb = A_BB_G.get_data()
        dados = bb.get("data") if isinstance(bb, dict) else None
        cam_paths = (bb.get("info", {}) if isinstance(bb, dict) else {}).get("primPaths", [])
        if dados is not None:
            for j in range(len(dados)):
                pid = pid_de_caminho(cam_paths[j]) if j < len(cam_paths) else None
                if pid:
                    r = dados[j]
                    arq_cx.write(f"{quadro},{pid},{float(r['x_min']):.1f},{float(r['y_min']):.1f},{float(r['x_max']):.1f},"
                                 f"{float(r['y_max']):.1f},{float(r['occlusionRatio']):.3f}\n")
    rps_ativos(False)
    for pid, (x, y, z, w) in reais.items():
        arq_gab.write(f"{quadro},{t:.4f},{pid},{x:.3f},{y:.3f},{z:.3f},{w:.2f}\n")
    arq_q.write(f"{quadro},{t:.4f},{C[0]:.3f},{C[1]:.3f},{C[2]:.3f},{alvo[0]:.3f},{alvo[1]:.3f},{alvo[2]:.3f},{hf:.2f},"
                f"{inclinacao_graus(C, alvo):.2f},{dpos[0]:.3f},{dpos[1]:.3f},{dpos[2]:.3f},{rumo:.2f},{incl:.2f},{vh:.2f},"
                f"{Cg[0]:.3f},{Cg[1]:.3f},{Cg[2]:.3f},{centro[0]:.3f},{centro[1]:.3f},"
                f"{(desvio[0] if desvio is not None else 0):.3f},{(desvio[1] if desvio is not None else 0):.3f},{sub},{dtr:.3f}\n")
    quadro += 1
    if quadro % 24 == 0 or heroi:
        el = time.time() - t_loop0 - t_avanco
        log(f"quadro {quadro} t={t:.2f}s; {el / quadro:.2f} s/quadro (render {t_render / quadro:.2f}); incl {inclinacao_graus(C, alvo):.1f} "
            f"graus; hfov {hf:.1f}; drone {np.round(dpos, 1).tolist()}{' HEROI' if heroi else ''}")
    world.step(render=True)

GRAVADOR.shutdown(wait=True)
for f in [arq_q, arq_gab] + ([arq_cx] if arq_cx else []):
    f.close()
el = time.time() - t_loop0
pico = None
for arq in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
    try:
        pico = int(Path(arq).read_text().strip()) / 2 ** 30
        break
    except Exception:  # noqa: BLE001
        pass
meta = {
    "simulador": "isaac-sim 5.1.0", "renderer": "RaytracedLighting (RTX tempo real)", "config_rtx": CONFIG_RTX,
    "anti_aliasing": args.anti_aliasing, "rt_subframes": args.rt_subframes, "rt_subframes_heroi": args.rt_subframes_heroi,
    "fps": args.fps, "quadros": quadro, "t_inicio": T0, "t_fim": T1, "instantes": instantes, "herois": herois,
    "cinegrafista": K_CINE.para_dict(), "gimbal": None if CAM_GIM is None else K_GIM.para_dict(),
    "pessoas": list(PERS), "grupo": GRUPO, "hdri": args.hdri, "ceu_intensidade": args.ceu_intensidade,
    "sol": {"elevacao": args.sol_elevacao, "azimute": args.sol_azimute, "intensidade": args.sol_intensidade},
    "materiais": args.materiais, "fatores_uv": LADRILHO, "tempo_total_script_s": time.time() - T_INI_SCRIPT,
    "tempo_laco_s": el, "tempo_avanco_s": t_avanco, "tempo_render_s": t_render,
    "s_por_quadro_gravado": (el - (t_avanco or 0)) / max(quadro, 1), "s_render_por_quadro": t_render / max(quadro, 1),
    "pico_memoria_cgroup_gib": pico, "desvio_final_grupo_m": None if desvio is None else np.round(desvio, 3).tolist(),
    "avisos": AVISOS, "args": vars(args), "plano": args.plano,
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
log(f"FIM: {quadro} quadros; laco {el / 60:.1f} min (avanco {t_avanco or 0:.0f} s, render {t_render:.0f} s); "
    f"pico cgroup {pico if pico is None else round(pico, 2)} GiB; saida {OUT}")
(OUT / "log_sim.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
app.close()
