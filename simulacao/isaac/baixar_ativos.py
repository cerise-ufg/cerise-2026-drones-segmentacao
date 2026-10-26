"""Espelho local PARCIAL dos ativos do Isaac Sim 5.1 usados na simulacao (para rodar com --network none).

Rodar SOMENTE depois do aceite da licenca do Isaac Sim (os ativos sao cobertos pela mesma licenca).
Fonte: servidor publico oficial configurado na imagem (OMNI_SERVER).
  python3 baixar_ativos.py --destino ~/cerise2026/drones/sim/isaac_ativos --listar      # so tamanhos
  python3 baixar_ativos.py --destino ~/cerise2026/drones/sim/isaac_ativos               # baixa (~0,5-0,7 GB)
Depois: ATIVOS_LOCAIS=~/cerise2026/drones/sim/isaac_ativos bash rodar_isaac.sh ...
"""
import argparse
import os
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

B = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
R = "Assets/Isaac/5.1/"
PREFIXOS = [
    "Isaac/People/Characters/Biped_Setup.usd",
    "Isaac/People/Characters/biped_demo/",          # esqueleto referenciado (payload) pelo Biped_Setup.usd
    "Isaac/People/Characters/filter.json",
    "Isaac/People/Animations/",
    "Isaac/People/Characters/F_Business_02/",
    "Isaac/People/Characters/F_Medical_01/",
    "Isaac/People/Characters/M_Medical_01/",
    "Isaac/People/Characters/female_adult_police_01_new/",
    "Isaac/People/Characters/female_adult_police_02/",
    "Isaac/People/Characters/female_adult_police_03_new/",
    "Isaac/People/Characters/male_adult_construction_01_new/",
    "Isaac/People/Characters/male_adult_construction_03/",
    "Isaac/People/Characters/male_adult_construction_05_new/",
    "Isaac/People/Characters/male_adult_police_04/",
    "Isaac/People/Characters/original_male_adult_construction_02/",
    "Isaac/People/Characters/original_male_adult_medical_01/",
    "Isaac/Robots/IsaacSim/Quadcopter/",
]
OPCIONAIS = ["NVIDIA/Materials/Base/Natural/"]  # --materiais biblioteca
# --closeup (isaac/closeup_isaac.py): cupula de ceu HDRI (Poly Haven, CC0, hospedada pela NVIDIA) e materiais MDL do chao
CLOSEUP = [
    "NVIDIA/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",
    "NVIDIA/Assets/Skies/Clear/noon_grass_4k.hdr",
    "NVIDIA/Materials/Base/Natural/Grass_Countryside.mdl",
    "NVIDIA/Materials/Base/Natural/Grass_Countryside/",
    "NVIDIA/Materials/Base/Natural/Grass_Cut.mdl",
    "NVIDIA/Materials/Base/Natural/Grass_Cut/",
    "NVIDIA/Materials/Base/Natural/Dirt.mdl",
    "NVIDIA/Materials/Base/Natural/Dirt/",
    "NVIDIA/Materials/Base/Natural/Asphalt.mdl",
    "NVIDIA/Materials/Base/Natural/Asphalt/",
    "NVIDIA/Materials/Base/Masonry/Concrete_Smooth.mdl",
    "NVIDIA/Materials/Base/Masonry/Concrete_Smooth/",
]


def listar(prefixo):
    objs, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefixo}
        if token:
            q["continuation-token"] = token
        x = ET.fromstring(urllib.request.urlopen(B + "?" + urllib.parse.urlencode(q), timeout=60).read())
        objs += [(c.find("s:Key", NS).text, int(c.find("s:Size", NS).text)) for c in x.findall("s:Contents", NS)]
        if x.find("s:IsTruncated", NS).text != "true":
            return objs
        token = x.find("s:NextContinuationToken", NS).text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", required=True)
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--com_materiais", action="store_true")
    ap.add_argument("--closeup", action="store_true", help="so os extras do close-up (HDRI + materiais do chao, ~0,15 GB)")
    a = ap.parse_args()
    dest = Path(os.path.expanduser(a.destino))
    pref = CLOSEUP if a.closeup else PREFIXOS + (OPCIONAIS if a.com_materiais else [])
    objs = []
    for p in pref:
        objs += [o for o in listar(R + p) if "/.thumbs/" not in o[0]]
    tot = sum(s for _, s in objs)
    print(f"{len(objs)} arquivos, {tot / 1e9:.2f} GB")
    if a.listar:
        return
    for i, (k, s) in enumerate(objs):
        alvo = dest / k
        if alvo.exists() and alvo.stat().st_size == s:
            continue
        alvo.parent.mkdir(parents=True, exist_ok=True)
        tmp = alvo.with_suffix(alvo.suffix + ".part")
        urllib.request.urlretrieve(B + urllib.parse.quote(k), tmp)
        tmp.rename(alvo)
        if i % 50 == 0:
            print(f"{i}/{len(objs)} {k}", flush=True)
    print("ok ->", dest / R)


if __name__ == "__main__":
    sys.exit(main())
