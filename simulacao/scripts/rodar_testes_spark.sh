#!/bin/bash
# Testes do cerise_sim no Spark (conteiner cerise26_drones_sim, so CPU). Uso: bash scripts/rodar_testes_spark.sh
BASE=~/cerise2026/drones
sudo docker run --rm --name cerise26_drones_testes --cpus 2 -u $(id -u):$(id -g) -e HOME=/w/home \
  -e PYTHONDONTWRITEBYTECODE=1 -v $BASE:/w cerise26_drones_sim:latest \
  bash -c "cd /w/sim && nice -n 19 python -m pytest -v -p no:cacheprovider tests"
