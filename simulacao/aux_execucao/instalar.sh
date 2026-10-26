#!/bin/bash
# instala o pacote de simulacao em ~/cerise2026/drones/{sim,cerise_drones,dados} e constroi a imagem de contagem
set -e
B=~/cerise2026/drones
mkdir -p $B/sim/tmp/x && cd $B/sim/tmp/x && rm -rf simulacao cerise_drones dados
tar -xzf $B/sim/tmp/sim.tgz
rsync -a simulacao/ $B/sim/
rsync -a cerise_drones/ $B/cerise_drones/
rsync -a dados/ $B/dados/
chmod +x $B/sim/isaac/*.sh $B/sim/gazebo/*.sh $B/sim/scripts/*.sh
cd $B/sim
echo "free:"; free -g | head -2; uptime
sudo nice -n 19 docker build -f Dockerfile.contagem -t cerise26_drones_sim:latest . > $B/sim/tmp/build_contagem.log 2>&1; echo "build rc=$?"
tail -n 5 $B/sim/tmp/build_contagem.log
sudo docker images cerise26_drones_sim
