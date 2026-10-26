#!/bin/bash
IMG=nvcr.io/nvidia/isaac-sim:5.1.0
OUT=~/cerise2026/drones/sim/isaac_inspecao
mkdir -p $OUT; chmod 777 $OUT
cp ~/cerise2026/drones/sim/tmp/inspect_in.sh $OUT/
sudo docker image inspect $IMG --format 'Arch={{.Architecture}} Os={{.Os}} Size={{.Size}} User={{.Config.User}} WorkDir={{.Config.WorkingDir}} Entrypoint={{json .Config.Entrypoint}} Cmd={{json .Config.Cmd}}' > $OUT/00_image_inspect.txt
sudo docker image inspect $IMG --format '{{json .Config.Env}}' | tr ',' '\n' >> $OUT/00_image_inspect.txt
sudo docker image inspect $IMG --format '{{json .Config.Labels}}' | tr ',' '\n' >> $OUT/00_image_inspect.txt
cat $OUT/00_image_inspect.txt | head -30
sudo nice -n 19 docker run --rm --name cerise26_isaac_inspecao --network none --cpus 2 --entrypoint bash -u 0 -v $OUT:/out $IMG /out/inspect_in.sh
sudo chown -R $(id -u):$(id -g) $OUT
wc -l $OUT/*.txt
