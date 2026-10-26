#!/bin/bash
# remove os conteineres temporarios desta tarefa (cerise26_isaac_* e cerise26_drones_* parados) e confere os servicos
echo "conteineres temporarios:"
sudo docker ps -a --format '{{.Names}}\t{{.Status}}' | grep -E '^cerise26_(isaac|drones)_' || echo "(nenhum)"
for c in $(sudo docker ps -a --filter status=exited --format '{{.Names}}' | grep -E '^cerise26_(isaac|drones)_'); do
  sudo docker rm "$c" >/dev/null && echo "removido $c"
done
echo "--- conteineres agora"
sudo docker ps -a --format '{{.Names}}\t{{.Status}}'
echo "--- servidor 3D (8095)"
curl -s -m 10 localhost:8095/api/v1/health; echo
echo "--- producao"
for p in 8000 8001 8002; do echo "porta $p /health: $(curl -s -m 10 -o /dev/null -w '%{http_code}' localhost:$p/health)"; done
echo "qdrant 6333: $(curl -s -m 10 -o /dev/null -w '%{http_code}' localhost:6333/)"
echo "vania-brain 8080: $(curl -s -m 10 -o /dev/null -w '%{http_code}' localhost:8080/)"
echo "ollama 11434: $(curl -s -m 10 localhost:11434/api/version)"
free -g | head -2
