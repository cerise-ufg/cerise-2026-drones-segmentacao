#!/bin/bash
# Plano B no Spark (host): checa memoria/braco_gen, (re)gera o mundo e roda o conteiner cerise26_gzpx4.
# Uso: bash ~/cerise2026/drones/sim/gazebo/rodar_gazebo.sh gz_run01 [reduzido] [fps]
set -euo pipefail
RUN=${1:?uso: rodar_gazebo.sh <nome_execucao> [cenario] [fps]}
CEN=${2:-reduzido}
FPS=${3:-5}
BASE=~/cerise2026/drones/sim
DISP=$(free -g | awk '/^Mem:/{print $7}')
[ "$DISP" -ge "${MIN_LIVRE_INICIO_GB:-24}" ] || { echo "memoria available ${DISP} GB < ${MIN_LIVRE_INICIO_GB:-24} GB" >&2; exit 3; }
if sudo docker ps --format '{{.Names}}' | grep -qx braco_gen; then
  [ "${IGNORAR_BRACO:-0}" = "1" ] || { echo "braco_gen ainda rodando; use IGNORAR_BRACO=1 para forcar" >&2; exit 4; }
fi
sudo docker image inspect cerise26_gzpx4:latest >/dev/null 2>&1 || { echo "imagem cerise26_gzpx4 ausente: construa com o Dockerfile" >&2; exit 5; }
# mundo (so numpy): usa a imagem de deteccao da frente de drones
sudo docker run --rm -u $(id -u):$(id -g) -v $BASE:/w cerise26_drones:latest \
  python /w/gazebo/gerar_mundo.py --cenario /w/cenario/$CEN --saida /w/gazebo/mundo_$CEN --fps $FPS
mkdir -p $BASE/saidas/$RUN
sudo docker rm -f cerise26_gzpx4_$RUN >/dev/null 2>&1 || true
sudo docker run -d --name cerise26_gzpx4_$RUN --gpus all --cpus "${CPUS:-8}" --cpu-shares 256 \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e CENARIO=/cenario/$CEN -e FPS=$FPS -e DURACAO="${DURACAO:-0}" \
  -v $BASE:/sim:ro -v $BASE/cenario:/cenario:ro -v $BASE/gazebo/mundo_$CEN:/mundo:ro -v $BASE/saidas/$RUN:/saida \
  cerise26_gzpx4:latest bash /sim/gazebo/rodar_sim_gz.sh
setsid nohup bash $BASE/isaac/vigia_memoria.sh cerise26_gzpx4_$RUN "${MIN_LIVRE_GB:-10}" > $BASE/saidas/$RUN.vigia.log 2>&1 < /dev/null &
echo "conteiner cerise26_gzpx4_$RUN iniciado; logs em $BASE/saidas/$RUN/logs"
