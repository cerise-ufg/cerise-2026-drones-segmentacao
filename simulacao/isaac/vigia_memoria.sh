#!/bin/bash
# Vigia de memoria (memoria unificada do GB10): le a cada 2 s a memoria "available" do host (registra a cada 10 s,
# com o uso do conteiner) e MATA o conteiner (docker kill) se ela cair abaixo do minimo (protege a producao).
# Antes (26/10) lia a cada 10 s e usava "docker stop -t 20": lento demais para um pico de > 30 GB em ~11 s.
# Uso: bash vigia_memoria.sh <conteiner> [min_livre_gb]
CONT=$1
MIN_GB=${2:-10}
echo "inicio $(date -Is) conteiner=$CONT min_livre=${MIN_GB}GB"
echo "data,available_MB,used_MB,conteiner_mem"
MIN_VISTO=999999
N=0
while sudo docker ps --format '{{.Names}}' | grep -qx "$CONT"; do
  read -r USED AVAIL < <(free -m | awk '/^Mem:/{print $3, $7}')
  [ "$AVAIL" -lt "$MIN_VISTO" ] && MIN_VISTO=$AVAIL
  if [ "$AVAIL" -lt $((MIN_GB * 1024)) ]; then
    echo "ALERTA $(date -Is): available ${AVAIL} MB < ${MIN_GB} GB -> matando $CONT"
    sudo docker kill "$CONT"
    break
  fi
  if [ $((N % 5)) -eq 0 ]; then
    CM=$(sudo docker stats --no-stream --format '{{.MemUsage}}' "$CONT" 2>/dev/null | cut -d/ -f1 | tr -d ' ')
    echo "$(date +%H:%M:%S),$AVAIL,$USED,$CM"
  fi
  N=$((N + 1))
  sleep 2
done
echo "fim $(date -Is); menor available observado: ${MIN_VISTO} MB"
