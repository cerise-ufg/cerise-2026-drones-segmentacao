#!/bin/bash
sudo docker manifest inspect nvcr.io/nvidia/isaac-sim:5.1.0 2>&1 | head -80
echo "--- root docker config (auths keys only)"
sudo python3 -c "import json;d=json.load(open('/root/.docker/config.json'));print(list(d.get('auths',{}).keys()))" 2>&1
ls ~/.docker 2>/dev/null && python3 -c "import json;d=json.load(open('$HOME/.docker/config.json'));print(list(d.get('auths',{}).keys()))" 2>&1
