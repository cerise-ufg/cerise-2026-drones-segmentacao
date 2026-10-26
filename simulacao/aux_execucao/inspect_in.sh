#!/bin/bash
# roda DENTRO do container (entrypoint bash; o app Isaac Sim NAO e iniciado)
O=/out
cd /isaac-sim
{ echo "== VERSION"; cat VERSION 2>/dev/null; echo; echo "== ls /isaac-sim"; ls -la; echo "== apps"; ls apps; echo "== whoami"; id; echo "== env"; env | sort | grep -v -i token; } > $O/01_raiz.txt 2>&1
{ for d in exts extsDeprecated extscache extsUser kit/exts kit/extscore extsPhysics; do echo "== $d"; ls $d 2>/dev/null; done; } > $O/02_extensoes.txt 2>&1
{ echo "== asset_root nas configs"; grep -rn -i "asset_root\|assets_root\|Assets/Isaac" apps/*.kit 2>/dev/null | head -40;
  for f in $(ls -d exts*/isaacsim.storage.native* extscache/isaacsim.storage.native* 2>/dev/null); do grep -rn -i "asset_root\|s3\|omniverse-content" $f/config 2>/dev/null | head -20; done; } > $O/03_asset_root.txt 2>&1
{ echo "== usd locais (people/character/quad/drone)"; find / -xdev \( -iname "*.usd" -o -iname "*.usda" -o -iname "*.usdz" -o -iname "*.usdc" \) 2>/dev/null | grep -i -E "people|character|biped|quad|crazyflie|drone|iris|x500" | head -50;
  echo "== total de arquivos usd no container"; find / -xdev \( -iname "*.usd" -o -iname "*.usda" -o -iname "*.usdc" -o -iname "*.usdz" \) 2>/dev/null | wc -l;
  echo "== mdl"; find / -xdev -iname "*.mdl" 2>/dev/null | wc -l; find / -xdev -iname "*.mdl" 2>/dev/null | grep -i -E "omnipbr|omnisurface|grass|ground|concrete|asphalt|dirt|soil" | head -40; } > $O/04_ativos_locais.txt 2>&1
{ echo "== python"; ls kit/python/bin 2>/dev/null; kit/python/bin/python3 --version 2>&1;
  for m in numpy PIL cv2 scipy imageio shapely pyproj; do echo "-- $m"; find / -xdev -maxdepth 8 -type d -name "$m" -path "*site-packages*" 2>/dev/null | head -3; find / -xdev -maxdepth 8 -type d -name "$m" -path "*pip_prebundle*" 2>/dev/null | head -3; done; } > $O/05_python.txt 2>&1
{ for pat in "omni.anim.people*" "isaacsim.replicator.agent*" "omni.anim.navigation*" "omni.anim.graph*" "omni.anim.retarget*" "isaacsim.anim*" "omni.replicator.core*" "isaacsim.replicator*" "omni.metropolis*" "isaacsim.asset.gen*"; do
    for d in $(ls -d exts*/$pat kit/exts*/$pat 2>/dev/null); do echo "=== $d"; ls $d; cat $d/config/extension.toml 2>/dev/null | head -40; done; done; } > $O/06_ext_pessoas.txt 2>&1
{ for d in $(ls -d exts*/omni.anim.people* exts*/isaacsim.replicator.agent* 2>/dev/null); do echo "=== $d"; find $d -name "*.py" | head -80; find $d \( -name "*.yaml" -o -name "*.txt" -o -name "*.md" \) | head -40; done; } > $O/07_people_arquivos.txt 2>&1
{ for d in $(ls -d exts*/omni.anim.people* exts*/isaacsim.replicator.agent* exts*/omni.anim.navigation* 2>/dev/null); do echo "=== $d"; grep -rn -E "GoTo|LookAround|Idle|command_file|navmesh|NavMesh|start_navmesh_baking|asset_root|Isaac/People|Biped_Setup|character_root|robot_root" --include=*.py $d 2>/dev/null | head -80; done; } > $O/08_people_api.txt 2>&1
{ for d in $(ls -d exts*/omni.replicator.core* 2>/dev/null); do echo "=== $d"; grep -rln "bounding_box_2d_tight" $d | head; grep -rn "\"camera_params\"\|'camera_params'" $d --include=*.py | head -5; done; } > $O/09_replicator.txt 2>&1
{ echo "== standalone_examples"; find standalone_examples -maxdepth 3 | grep -i -E "replicator|people|agent|anim|sdg|camera" | head -60; echo "== scripts sh"; ls *.sh; echo "== runheadless"; cat runheadless.sh 2>/dev/null | head -40; echo "== python.sh"; cat python.sh | head -60; } > $O/10_exemplos_scripts.txt 2>&1
{ echo "== docs/licencas"; ls /isaac-sim/*LICENSE* /isaac-sim/*EULA* /isaac-sim/docs 2>/dev/null; find / -xdev -maxdepth 3 -iname "*eula*" 2>/dev/null | head; grep -rn -i "ACCEPT_EULA\|PRIVACY_CONSENT" /isaac-sim/*.sh /entrypoint* /usr/local/bin 2>/dev/null | head -20; } > $O/11_eula.txt 2>&1
echo ok
