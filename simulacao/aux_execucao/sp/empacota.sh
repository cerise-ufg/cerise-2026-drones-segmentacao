#!/bin/bash
cd ~/cerise2026/drones/sim/saidas
tar czf ~/cerise2026/drones/sim/tmp/leves.tgz run01/meta.json run01/gabarito_pessoas.csv run01/*/poses.csv run01/D?/caixas_gt.csv \
  run02a30/meta.json run02a30/gabarito_pessoas.csv run02a30/*/poses.csv run02a30/D?/caixas_gt.csv
ls -la ~/cerise2026/drones/sim/tmp/leves.tgz
