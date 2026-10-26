#!/bin/bash
# tamanho comprimido da variante arm64
sudo docker manifest inspect nvcr.io/nvidia/isaac-sim@sha256:2f8f13549d9eca03393d5b34f49f5b56bfa7b3b69556c319c98ff7a6273b6bde > ~/cerise2026/drones/sim/tmp/manifest_arm64.json 2>&1
python3 - <<'PY'
import json,os
d=json.load(open(os.path.expanduser('~/cerise2026/drones/sim/tmp/manifest_arm64.json')))
t=sum(l['size'] for l in d['layers']); print('camadas',len(d['layers']),'comprimido GB %.2f'%(t/1e9))
PY
