#!/bin/bash
# amostra utilizacao de GPU e carga de CPU por N segundos
N=${1:-60}
for i in $(seq 1 $((N/5))); do
  echo "$(date +%T) gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader) load=$(cut -d' ' -f1 /proc/loadavg) avail=$(free -g | awk '/Mem/{print $7}')G"
  sleep 5
done
