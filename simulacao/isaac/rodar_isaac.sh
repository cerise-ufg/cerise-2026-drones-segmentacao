#!/bin/bash
# Roda a simulacao do Isaac Sim 5.1 (headless, sem tela fisica, sem streaming) no DGX Spark.
#
# Licenca "NVIDIA Isaac Sim Additional Software and Materials License" aceita em 26/10/2026 (~11:10),
# com PRIVACY_CONSENT=N (sem telemetria). Exemplos:
#   ACCEPT_EULA=Y PRIVACY_CONSENT=N bash ~/cerise2026/drones/sim/isaac/rodar_isaac.sh smoke --smoke
#   ACCEPT_EULA=Y PRIVACY_CONSENT=N bash ~/cerise2026/drones/sim/isaac/rodar_isaac.sh run01 --cenario /cenario/reduzido --fps 5
# Argumentos depois do nome vao para isaac/sim_ufg_isaac.py (duracao, fps, n_pessoas, largura, modo_pessoas...).
#
# Variaveis opcionais:
#   ATIVOS_LOCAIS=/caminho/espelho   espelho local dos ativos (isaac/baixar_ativos.py); SEM_REDE=1 -> --network none
#   CPUS=8                           limite de CPU do conteiner
#   MIN_LIVRE_INICIO_GB=30           memoria "available" minima para iniciar (job ~15-20 GB + folga de 10 GB)
#   MIN_LIVRE_GB=12                  o vigia MATA o conteiner se a memoria "available" cair abaixo disso (le a cada 2 s)
#   MEM_LIMITE=24g                   limite de memoria (cgroup) do conteiner
#   IGNORAR_BRACO=1                  permite iniciar com o braco_gen ainda rodando (nao recomendado)
#   SCRIPT=/sim/isaac/closeup_isaac.py  outro script do Isaac (padrao: /sim/isaac/sim_ufg_isaac.py), ex.: o close-up
set -euo pipefail
NOME=${1:?uso: rodar_isaac.sh <nome_execucao> [args do sim_ufg_isaac.py]}
shift
BASE=~/cerise2026/drones/sim
IMG=nvcr.io/nvidia/isaac-sim:5.1.0
CONT=cerise26_isaac_${NOME}
ACCEPT_EULA=${ACCEPT_EULA:-}
PRIVACY_CONSENT=${PRIVACY_CONSENT:-N}

if [ "$ACCEPT_EULA" != "Y" ]; then
  echo "ERRO: ACCEPT_EULA nao e 'Y'. A licenca do Isaac Sim precisa ser aceita antes de rodar." >&2
  exit 2
fi
DISP=$(free -g | awk '/^Mem:/{print $7}')
if [ "$DISP" -lt "${MIN_LIVRE_INICIO_GB:-30}" ]; then
  echo "ERRO: memoria available ${DISP} GB < ${MIN_LIVRE_INICIO_GB:-30} GB (job ~15-20 GB + folga de 10 GB). Espere ou reduza resolucao/pessoas." >&2
  exit 3
fi
if sudo docker ps --format '{{.Names}}' | grep -qx braco_gen; then
  echo "AVISO: o conteiner braco_gen ainda esta rodando (CPU ocupada)." >&2
  [ "${IGNORAR_BRACO:-0}" = "1" ] || { echo "Saindo. Use IGNORAR_BRACO=1 para forcar." >&2; exit 4; }
fi
sudo docker rm -f "$CONT" >/dev/null 2>&1 || true

C=$BASE/isaac_cache
mkdir -p $C/main $C/computecache $C/logs $C/config $C/data $C/pkg $BASE/saidas/$NOME
sudo chown -R 1234:1234 $C $BASE/saidas/$NOME     # o usuario do conteiner e isaac-sim (uid 1234)

EXTRA=()
if [ -n "${ATIVOS_LOCAIS:-}" ]; then
  EXTRA+=(-v "$ATIVOS_LOCAIS":/ativos:ro -e ISAAC_ASSETS_ROOT=/ativos/Assets/Isaac/5.1)
  # SEM_REDE=1: contêiner sem rede (só o espelho local); padrão: rede bridge sem portas publicadas
  [ "${SEM_REDE:-0}" = "1" ] && EXTRA+=(--network none)
fi
# PRIVACY_CONSENT=N desliga a telemetria (config/privacy.toml da imagem); OMNI_ENV_PRIVACY_CONSENT nunca e definido.
# --memory: limite de cgroup (parte de CPU da memoria unificada) para proteger a producao de um OOM global.
# (Em 26/10 a assadura da navmesh do cenario "completo" consumiu > 34 GB em ~11 s; o vigia, que lia a cada 10 s,
#  nao chegou a tempo e houve OOM global. Ver RELATORIO_SECAO_SIMULACAO.md.)
sudo docker run -d --name "$CONT" --gpus all --cpus "${CPUS:-8}" --cpu-shares 256 \
  --memory "${MEM_LIMITE:-24g}" --memory-swap "${MEM_LIMITE:-24g}" \
  -e ACCEPT_EULA="$ACCEPT_EULA" -e PRIVACY_CONSENT="$PRIVACY_CONSENT" \
  -e PYTHONUNBUFFERED=1 \
  "${EXTRA[@]}" \
  -v $C/main:/isaac-sim/.cache:rw \
  -v $C/computecache:/isaac-sim/.nv/ComputeCache:rw \
  -v $C/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
  -v $C/config:/isaac-sim/.nvidia-omniverse/config:rw \
  -v $C/data:/isaac-sim/.local/share/ov/data:rw \
  -v $C/pkg:/isaac-sim/.local/share/ov/pkg:rw \
  -v $BASE:/sim:ro \
  -v $BASE/cenario:/cenario:ro \
  -v $BASE/saidas/$NOME:/saida:rw \
  -u 1234:1234 \
  --entrypoint /isaac-sim/python.sh \
  $IMG "${SCRIPT:-/sim/isaac/sim_ufg_isaac.py}" --cenario /cenario/reduzido --saida /saida "$@"

echo "conteiner $CONT iniciado; saida em $BASE/saidas/$NOME"
setsid nohup bash $BASE/isaac/vigia_memoria.sh "$CONT" "${MIN_LIVRE_GB:-12}" > $BASE/saidas/$NOME.vigia.log 2>&1 < /dev/null &
echo "vigia de memoria: $BASE/saidas/$NOME.vigia.log ; log do app: sudo docker logs -f $CONT"
