#!/bin/bash
# Uso: bash run.sh <nome> <script.py> [args...]
# Roda um script de deteccao/ dentro do container cerise26_drones (GPU), destacado, com log em logs/<nome>.log
NAME=$1; shift
SCRIPT=$1; shift
BASE=~/cerise2026/drones
mkdir -p $BASE/logs $BASE/home $BASE/ultra_cfg
sudo docker rm -f cerise26_drones_$NAME >/dev/null 2>&1
sudo docker run -d --name cerise26_drones_$NAME --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -u $(id -u):$(id -g) -e HOME=/w/home -e MPLCONFIGDIR=/w/home/.mpl \
  -e YOLO_CONFIG_DIR=/w/ultra_cfg -e PYTHONUNBUFFERED=1 \
  -v $BASE:/w -v ~/cerise2026/hf_cache:/w/home/.cache/huggingface \
  cerise26_drones:latest \
  bash -c "cd /w && nice -n 10 python /w/deteccao/$SCRIPT $* > /w/logs/$NAME.log 2>&1"
echo "container cerise26_drones_$NAME iniciado; log: $BASE/logs/$NAME.log"
