#!/bin/bash
# Vigia de memoria (memoria unificada do GB10): registra a cada 10 s a memoria "available" do host e o uso
# do conteiner; PARA o conteiner se a "available" cair abaixo do minimo (protege a producao: vLLM etc.).
# Uso: bash vigia_memoria.sh <conteiner> [min_livre_gb]
CONT=$1
MIN_GB=${2:-10}
echo "inicio $(date -Is) conteiner=$CONT min_livre=${MIN_GB}GB"
echo "data,available_MB,used_MB,conteiner_mem"
MIN_VISTO=999999
while sudo docker ps --format '{{.Names}}' | grep -qx "$CONT"; do
  read -r USED AVAIL < <(free -m | awk '/^Mem:/{print $3, $7}')
  CM=$(sudo docker stats --no-stream --format '{{.MemUsage}}' "$CONT" 2>/dev/null | cut -d/ -f1 | tr -d ' ')
  echo "$(date +%H:%M:%S),$AVAIL,$USED,$CM"
  [ "$AVAIL" -lt "$MIN_VISTO" ] && MIN_VISTO=$AVAIL
  if [ "$AVAIL" -lt $((MIN_GB * 1024)) ]; then
    echo "ALERTA $(date -Is): available ${AVAIL} MB < ${MIN_GB} GB -> parando $CONT"
    sudo docker stop -t 20 "$CONT"
    break
  fi
  sleep 10
done
echo "fim $(date -Is); menor available observado: ${MIN_VISTO} MB"
