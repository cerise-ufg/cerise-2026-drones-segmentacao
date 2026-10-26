#!/bin/bash
set -e
B=~/cerise2026/drones
mkdir -p $B/sim/tmp/x && cd $B/sim/tmp/x && rm -rf simulacao cerise_drones dados
tar -xzf $B/sim/tmp/sim.tgz
rsync -a simulacao/ $B/sim/
rsync -a cerise_drones/ $B/cerise_drones/
rsync -a dados/ $B/dados/
chmod +x $B/sim/isaac/*.sh $B/sim/gazebo/*.sh $B/sim/scripts/*.sh
ls $B/sim $B/sim/cenario; du -sh $B/sim/cenario/*
