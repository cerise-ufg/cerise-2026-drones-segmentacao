#!/bin/bash
cd ~/cerise2026/drones/sim
export ACCEPT_EULA=Y PRIVACY_CONSENT=N ATIVOS_LOCAIS=/home/energiainteligente/cerise2026/drones/sim/isaac_ativos
for n in smoke2; do sudo docker rm $n >/dev/null 2>&1; done
sudo docker rm cerise26_isaac_smoke2 >/dev/null 2>&1
bash isaac/rodar_isaac.sh smoke3 --smoke --duracao 8 --pessoas P059,P071,P063,P000,P030,P022 --t_inicio 75 --modo_pessoas cinematico --camera_debug P022
while sudo docker ps --format '{{.Names}}' | grep -qx cerise26_isaac_smoke3; do sleep 5; done
bash isaac/rodar_isaac.sh smoke4 --smoke --duracao 8 --pessoas P059,P071,P063,P000,P030,P022 --t_inicio 75 --camera_debug P022
while sudo docker ps --format '{{.Names}}' | grep -qx cerise26_isaac_smoke4; do sleep 5; done
echo FIM_SMOKE34
