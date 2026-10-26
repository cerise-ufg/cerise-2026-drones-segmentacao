#!/bin/bash
# uso: bash dump_usd.sh <arquivo.usd relativo a /sim>
sudo docker run --rm --name cerise26_isaac_dumpusd --network none --memory 4g -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=N \
  -v ~/cerise2026/drones/sim:/sim:ro -u 1234:1234 --entrypoint bash nvcr.io/nvidia/isaac-sim:5.1.0 -c '
L=$(ls -d /isaac-sim/extscache/omni.usd.libs-*)
export PYTHONPATH=$L:$PYTHONPATH LD_LIBRARY_PATH=$L/bin:$LD_LIBRARY_PATH
/isaac-sim/kit/python/bin/python3 /sim/closeup/dump_usd.py "$0"' "$1" 2>&1 | tail -80
