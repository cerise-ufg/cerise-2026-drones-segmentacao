#!/bin/bash
IMG=nvcr.io/nvidia/isaac-sim:5.1.0
echo '$ sudo docker manifest inspect nvcr.io/nvidia/isaac-sim:5.1.0 | grep architecture'
sudo docker manifest inspect $IMG | grep '"architecture"'
echo '$ sudo docker images nvcr.io/nvidia/isaac-sim'
sudo docker images $IMG
echo "\$ sudo docker image inspect $IMG --format 'Arch Size User Entrypoint MIN_DRIVER OMNI_SERVER'"
sudo docker image inspect $IMG --format '{{.Architecture}} {{.Size}} {{.Config.User}} {{json .Config.Entrypoint}}'
sudo docker image inspect $IMG --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E 'MIN_DRIVER|OMNI_SERVER'
echo '$ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader'
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo '$ sudo docker run --rm --network none --entrypoint bash isaac-sim:5.1.0 -c "cat VERSION; ls extscache | grep -E people|agent.core|navigation.core|replicator.core; tail -3 runheadless.sh"   # app NAO iniciado'
sudo docker run --rm --network none --entrypoint bash $IMG -c 'cd /isaac-sim; cat VERSION; echo; ls extscache | grep -E "omni.anim.people|replicator.agent.core|anim.navigation.core|omni.replicator.core"; sed -n 3,5p runheadless.sh' 2>&1
