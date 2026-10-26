#!/bin/bash
sudo docker run -i --rm --name cerise26_drones_chk -e YOLO_OFFLINE=1 -e YOLO_CONFIG_DIR=/tmp cerise26_drones_sim:latest python - 2>/dev/null <<'PY'
import inspect
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
print(inspect.getsource(BYTETracker.__init__))
src = inspect.getsource(BYTETracker.update)
print(src[:6000])
print(inspect.getsource(BOTSORT.__init__))
PY
