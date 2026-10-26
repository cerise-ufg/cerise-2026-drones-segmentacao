#!/bin/bash
# Executado DENTRO do conteiner cerise26_gzpx4: Gazebo servidor headless (EGL) + N instancias PX4 SITL
# anexadas aos x500 ja presentes no mundo (PX4_GZ_MODEL_NAME) + captura das cameras + missoes via MAVSDK.
# Nunca usa DISPLAY / GUI.
set -uo pipefail
unset DISPLAY
MUNDO=/mundo
SAIDA=/saida
mkdir -p $SAIDA/logs
INST=$MUNDO/px4_instancias.json
N=$(python3 -c "import json;print(len(json.load(open('$INST'))['instancias']))")
DUR=${DURACAO:-0}
[ "$DUR" = "0" ] && DUR=$(python3 -c "import json;print(json.load(open('$INST'))['duracao_s'])")

echo "[gz] iniciando servidor (headless rendering)"
gz sim -s -r --headless-rendering -v 3 $MUNDO/mundo_ufg.sdf > $SAIDA/logs/gz_server.log 2>&1 &
GZ_PID=$!
for i in $(seq 1 120); do
  gz topic -l 2>/dev/null | grep -q "/world/ufg/clock" && break
  sleep 1
done
echo "[gz] mundo no ar"

for k in $(seq 0 $((N - 1))); do
  read -r I MODELO < <(python3 -c "import json;d=json.load(open('$INST'))['instancias'][$k];print(d['instancia_px4'], d['modelo_gz'])")
  W=$SAIDA/logs/px4_$I; mkdir -p $W
  ( cd $W && PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_NAME=$MODELO PX4_GZ_WORLD=ufg PX4_GZ_STANDALONE=1 \
      /px4/build/px4_sitl_default/bin/px4 -i $I -d /px4/build/px4_sitl_default/etc > $W/px4.log 2>&1 ) &
  echo "[px4] instancia $I -> $MODELO"
done

python3 /sim/gazebo/capturar.py --mundo $MUNDO --cenario ${CENARIO:-/cenario/reduzido} --saida $SAIDA \
  --fps ${FPS:-5} --duracao "$DUR" > $SAIDA/logs/captura.log 2>&1 &
CAP_PID=$!
python3 /sim/gazebo/rodar_missoes.py --instancias $INST --cenario ${CENARIO:-/cenario/reduzido} > $SAIDA/logs/missoes.log 2>&1
echo "[missoes] fim (rc=$?)"
wait $CAP_PID
echo "[captura] fim"
kill $GZ_PID 2>/dev/null
pkill -f "px4_sitl_default/bin/px4" 2>/dev/null   # so dentro deste conteiner
exit 0
