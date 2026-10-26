#!/bin/bash
sudo docker run --rm -u $(id -u):$(id -g) -v ~/cerise2026/drones:/w cerise26_drones:latest bash -c 'cp $(python -c "import ultralytics,os;print(os.path.dirname(ultralytics.__file__))")/cfg/datasets/VisDrone.yaml /w/VisDrone_ultralytics.yaml' > /dev/null 2>&1
sed -n '/download:/,$p' ~/cerise2026/drones/VisDrone_ultralytics.yaml
