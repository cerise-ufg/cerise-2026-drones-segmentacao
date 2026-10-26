#!/bin/bash
cd ~/cerise2026/drones/sim
ls cenario/completo/ | tr '\n' ' '; echo
sudo docker rm cerise26_isaac_run01 cerise26_isaac_run02a30 >/dev/null 2>&1
free -g | awk '/^Mem:/{print "available GB:", $7}'
ACCEPT_EULA=Y PRIVACY_CONSENT=N ATIVOS_LOCAIS=/home/energiainteligente/cerise2026/drones/sim/isaac_ativos \
  bash isaac/rodar_isaac.sh run03completo --cenario /cenario/completo --fps 5 --sem_visao_geral 2>&1 | tail -2
date +%T
