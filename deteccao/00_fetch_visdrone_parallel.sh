#!/bin/bash
# Baixa os zips oficiais do VisDrone2019-DET (mesmas URLs do VisDrone.yaml do ultralytics, GitHub ultralytics/assets)
# com varias conexoes em paralelo (requisicoes HTTP Range), pois a rede do Spark esta compartilhada.
set -e
DEST=~/cerise2026/drones/datasets/VisDrone
mkdir -p $DEST/parts
BASE=https://github.com/ultralytics/assets/releases/download/v0.0.0
NP=${NP:-8}
for F in VisDrone2019-DET-val.zip VisDrone2019-DET-test-dev.zip VisDrone2019-DET-train.zip; do
  [ -f $DEST/$F ] && { echo "$F ja existe"; continue; }
  REAL=$(curl -sIL -o /dev/null -w '%{url_effective}' $BASE/$F)
  LEN=$(curl -sI "$REAL" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r')
  echo "$F: $LEN bytes"
  CH=$(( (LEN + NP - 1) / NP ))
  pids=()
  for i in $(seq 0 $((NP-1))); do
    S=$(( i*CH )); E=$(( S+CH-1 )); [ $E -ge $LEN ] && E=$((LEN-1))
    curl -sL --retry 5 -r $S-$E -o $DEST/parts/$F.$i "$REAL" & pids+=($!)
  done
  for p in ${pids[@]}; do wait $p; done
  cat $(for i in $(seq 0 $((NP-1))); do echo $DEST/parts/$F.$i; done) > $DEST/$F.tmp
  GOT=$(stat -c %s $DEST/$F.tmp)
  if [ "$GOT" = "$LEN" ]; then mv $DEST/$F.tmp $DEST/$F; rm -f $DEST/parts/$F.*; echo "$F OK ($GOT bytes)"; else echo "$F tamanho errado $GOT != $LEN"; exit 1; fi
done
ls -la $DEST
