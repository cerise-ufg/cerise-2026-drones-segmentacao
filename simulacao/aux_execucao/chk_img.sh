#!/bin/bash
cd ~/cerise2026/drones/sim/tmp; tail -c 600 pull.log; echo
free -g | head -2; uptime
sudo docker run --rm --entrypoint python cerise26_drones:latest -c "
import importlib
for m in ['numpy','scipy','shapely','pyproj','cv2','ultralytics','lap','pandas','matplotlib','pytest','filterpy']:
    try:
        x=importlib.import_module(m); print(m, getattr(x,'__version__','?'))
    except Exception as e: print(m,'FALTA',type(e).__name__)
"
ls ~/cerise2026/drones/deteccao ~/cerise2026/drones/resultados ~/cerise2026/drones/weights ~/cerise2026/drones/runs/*/weights 2>/dev/null
