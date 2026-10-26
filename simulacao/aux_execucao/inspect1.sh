#!/bin/bash
# Inspecao da imagem Isaac Sim 5.1.0 SEM iniciar o app (sem ACCEPT_EULA, sem GPU, sem rede)
IMG=nvcr.io/nvidia/isaac-sim:5.1.0
OUT=~/cerise2026/drones/sim/isaac_inspecao
mkdir -p $OUT
tail -n 3 ~/cerise2026/drones/sim/tmp/pull.log
sudo docker image inspect $IMG --format 'Arch={{.Architecture}} Os={{.Os}} Size={{.Size}} User={{.Config.User}} WorkDir={{.Config.WorkingDir}} Entrypoint={{json .Config.Entrypoint}} Cmd={{json .Config.Cmd}}' | tee $OUT/image_inspect.txt
sudo docker image inspect $IMG --format '{{json .Config.Env}}' | tr ',' '\n' >> $OUT/image_inspect.txt
R="sudo docker run --rm --network none --entrypoint bash $IMG -c"
$R 'cat /isaac-sim/VERSION 2>/dev/null; echo; ls -la /isaac-sim; echo; du -sh /isaac-sim 2>/dev/null; ls /isaac-sim/standalone_examples /isaac-sim/standalone_examples/* | head -80' > $OUT/ls_raiz.txt 2>&1
$R 'for d in /isaac-sim/exts /isaac-sim/extsDeprecated /isaac-sim/extscache /isaac-sim/extsUser /isaac-sim/kit/exts /isaac-sim/kit/extscore /isaac-sim/extsPhysics; do echo "== $d"; ls $d 2>/dev/null; done' > $OUT/ls_exts.txt 2>&1
wc -l $OUT/*.txt
grep -iE "anim|people|replicator|navigation|agent|drone|quad|pegasus|material|ros2|sensor|camera|storage|asset" $OUT/ls_exts.txt | sort | uniq | head -120
