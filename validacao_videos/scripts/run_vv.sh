#!/bin/bash
# Uso: [GPU=0] [CPUS=4] bash run_vv.sh <nome> <script.py> [args...]
# Roda um script de validacao_videos/scripts no container cerise26_drones (destacado, nice 19, log em logs/vv_<nome>.log).
# GPU=0 -> sem --gpus (etapas so de CPU). CPUS limita os nucleos do container (Spark compartilhado).
NAME=$1; shift
SCRIPT=$1; shift
BASE=~/cerise2026/drones
GPU=${GPU:-1}
CPUS=${CPUS:-4}
mkdir -p $BASE/logs $BASE/home $BASE/ultra_cfg $BASE/validacao_videos/resultados
GPUFLAG=""; [ "$GPU" = "1" ] && GPUFLAG="--gpus all"
sudo docker rm -f cerise26_drones_vv_$NAME >/dev/null 2>&1
sudo docker run -d --name cerise26_drones_vv_$NAME $GPUFLAG --cpus $CPUS --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -u $(id -u):$(id -g) -e HOME=/w/home -e MPLCONFIGDIR=/w/home/.mpl \
  -e YOLO_CONFIG_DIR=/w/ultra_cfg -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/w/pylib_mot:/w/validacao_videos/scripts \
  -e SEM_VIDEO=${SEM_VIDEO:-} -e OMP_NUM_THREADS=$CPUS -e OPENCV_NUM_THREADS=$CPUS \
  -v $BASE:/w cerise26_drones:latest \
  bash -c "cd /w/validacao_videos && nice -n 19 python /w/validacao_videos/scripts/$SCRIPT $* > /w/logs/vv_$NAME.log 2>&1"
echo "container cerise26_drones_vv_$NAME iniciado; log: $BASE/logs/vv_$NAME.log"
