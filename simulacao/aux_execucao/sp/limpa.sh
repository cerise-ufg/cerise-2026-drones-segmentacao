#!/bin/bash
for c in $(sudo docker ps -a --format '{{.Names}} {{.Status}}' | awk '$2=="Exited"{print $1}' | grep -E '^cerise26_(drones_|isaac_)'); do
  sudo docker rm "$c" >/dev/null && echo "removido $c"
done
pgrep -af vigia_memoria | grep -v pgrep || echo "sem vigias ativos"
echo "== conteineres restantes"; sudo docker ps -a --format '{{.Names}} {{.Status}}'
echo "== imagens"; sudo docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -E 'isaac|cerise26_drones'
echo "== saude"; curl -s -m 5 localhost:8095/api/v1/health | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["status"], "workers", d["workers"], d["worker_detail"][0]["state"])'
curl -s -m 5 -o /dev/null -w 'vllm-server %{http_code}\n' localhost:8000/v1/models
curl -s -m 5 -o /dev/null -w 'vania %{http_code}\n' localhost:8080/
curl -s -m 5 localhost:11434/api/version; echo
free -g | head -2; df -h ~ | tail -1; du -sh ~/cerise2026/drones/sim/saidas ~/cerise2026/drones/sim/tmp
