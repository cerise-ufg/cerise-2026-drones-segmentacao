#!/bin/bash
# Constroi a imagem cerise26_drones:latest no Spark
cd ~/cerise2026/drones/deteccao
sudo docker build -t cerise26_drones:latest . 2>&1 | tail -n 25
