#!/bin/bash
OUT=~/cerise2026/drones/sim/isaac_inspecao
cp ~/cerise2026/drones/sim/tmp/ref2_in.sh $OUT/
sudo nice -n 19 docker run --rm --name cerise26_isaac_inspecao --network none --cpus 2 --entrypoint bash -u 0 -v $OUT:/out nvcr.io/nvidia/isaac-sim:5.1.0 /out/ref2_in.sh > $OUT/12_navmesh_api.txt 2>&1
cat $OUT/12_navmesh_api.txt
