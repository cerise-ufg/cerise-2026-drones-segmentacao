#!/bin/bash
# uso: bash esperar.sh <conteiner> <padrao_no_log> [timeout_s=530]
# espera o padrao aparecer no log do conteiner (ou o conteiner sair) e mostra o progresso e o vigia
C=$1; P=$2; T=${3:-530}
timeout "$T" bash -c "until sudo docker logs $C 2>&1 | grep -q -- \"$P\"; do sleep 15; sudo docker ps --format '{{.Names}}' | grep -qx $C || break; done"
date
sudo docker ps -a --format '{{.Names}}\t{{.Status}}' | grep -x -- "$C.*" || true
sudo docker logs "$C" 2>&1 | grep -E '\[closeup|Traceback' | tail -8
V=~/cerise2026/drones/sim/saidas/${C#cerise26_isaac_}.vigia.log
tail -3 "$V"
free -g | head -2
