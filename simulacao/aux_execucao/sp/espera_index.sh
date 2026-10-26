#!/bin/bash
# espera o termino do chown/indice do dataset do braco em ~/braco (no maximo $1 s)
LIM=${1:-300}
T0=$(date +%s)
while true; do
  OCUP=$(pgrep -af "chown|build_index|status_dataset" | grep -v pgrep | grep -v espera_index | wc -l)
  DONO=$(stat -c %U ~/braco/dataset/000001.json)
  IDX=$(stat -c %Y ~/braco/dataset/index.csv 2>/dev/null || echo 0)
  AGORA=$(date +%s)
  echo "$(date +%T) ocupados=$OCUP dono=$DONO index_ha=$((AGORA-IDX))s"
  if [ "$OCUP" = "0" ] && [ $((AGORA-T0)) -ge 240 ]; then break; fi
  [ $((AGORA-T0)) -ge $LIM ] && break
  sleep 30
done
