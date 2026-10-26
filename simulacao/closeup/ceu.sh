#!/bin/bash
# processa os HDRIs (remove o sol) no conteiner de contagem (cv2), sem rede
sudo docker run --rm --name cerise26_drones_ceu --network none --memory 6g -u $(id -u):$(id -g) \
  -v ~/cerise2026/drones:/w cerise26_drones_sim:latest bash -c '
cd /w/sim
for H in Cloudy/kloofendal_48d_partly_cloudy_4k Clear/noon_grass_4k; do
  B=$(basename $H)
  python scripts/ceu_sem_sol.py isaac_ativos/Assets/Isaac/5.1/NVIDIA/Assets/Skies/$H.hdr closeup/ceu/${B}_sem_sol.hdr --json closeup/ceu/${B}_sem_sol.json | grep -E "irradiancia|fracao|elevacao"
done' 2>&1 | grep -v -E "^$|NOTE|docker run|====|NVIDIA|Copyright|license|Container|container|Use the|By pulling|https|WARNING|libraries|governed|SHMEM|insufficient"
