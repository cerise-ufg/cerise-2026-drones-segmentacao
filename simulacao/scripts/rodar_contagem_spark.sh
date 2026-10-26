#!/bin/bash
# Roda um script do pipeline de contagem no conteiner cerise26_drones_sim (GPU, FP32), destacado, com log.
# Uso: bash ~/cerise2026/drones/sim/scripts/rodar_contagem_spark.sh <nome> <script.py relativo a sim/> [args...]
#   ex.: bash scripts/rodar_contagem_spark.sh cont_run01 scripts/contar.py --execucao saidas/run01 \
#          --cenario cenario/reduzido --detector yolo --pesos /w/resultados/best_yolo11s_visdrone_pessoa_960.pt \
#          --saida resultados/contagem_run01.json --figura
# Antes: confira `free -g` (available >= 10 GB alem do job; o YOLO11s FP32 usa ~2-3 GB).
NOME=$1; shift
SCRIPT=$1; shift
BASE=~/cerise2026/drones
mkdir -p $BASE/sim/logs $BASE/home $BASE/ultra_cfg
DISP=$(free -g | awk '/^Mem:/{print $7}')
[ "$DISP" -ge 14 ] || { echo "memoria available ${DISP} GB < 14 GB; espere" >&2; exit 3; }
sudo docker rm -f cerise26_drones_$NOME >/dev/null 2>&1
sudo docker run -d --name cerise26_drones_$NOME --gpus all --ipc=host --cpus 4 \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -u $(id -u):$(id -g) -e HOME=/w/home -e MPLCONFIGDIR=/w/home/.mpl \
  -e YOLO_CONFIG_DIR=/w/ultra_cfg -e YOLO_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -v $BASE:/w -v ~/cerise2026/hf_cache:/w/home/.cache/huggingface \
  cerise26_drones_sim:latest \
  bash -c "cd /w/sim && nice -n 10 python $SCRIPT $* > /w/sim/logs/$NOME.log 2>&1"
echo "conteiner cerise26_drones_$NOME iniciado; log: $BASE/sim/logs/$NOME.log"
