#!/bin/bash
# uso: espera.sh <conteiner> <padrao_no_log> <timeout_s>  -> retorna quando o padrao aparece, o conteiner sai ou o tempo acaba
C=$1; P=$2; T=${3:-500}
T0=$(date +%s)
while [ $(( $(date +%s) - T0 )) -lt $T ]; do
  sudo docker logs "$C" 2>&1 | grep -q -- "$P" && break
  sudo docker ps --format '{{.Names}}' | grep -qx "$C" || break
  sleep 15
done
sudo docker ps -a --format '{{.Names}} {{.Status}}' | grep -x "$C.*" | head -1
sudo docker logs "$C" 2>&1 | grep -E 'cerise_sim|Traceback \(most|ERRO|FIM' | grep -v DEPURAR | tail -${4:-6} | cut -c1-260
V=~/cerise2026/drones/sim/saidas/${C#cerise26_isaac_}.vigia.log
[ -f $V ] && tail -1 $V
free -g | awk '/^Mem:/{print "available GB:", $7}'; date +%T
