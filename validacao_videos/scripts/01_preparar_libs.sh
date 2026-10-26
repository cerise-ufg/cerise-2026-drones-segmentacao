#!/bin/bash
# Instala, SEM mexer na imagem cerise26_drones:latest, as bibliotecas extras da validacao em video
# num diretorio montado (/w/pylib_mot, via PYTHONPATH -- ver run_vv.sh), a partir do PyPI:
#   lap        associacao linear usada pelos rastreadores ByteTrack/BoT-SORT do ultralytics
#   motmetrics MOTA/IDF1 de referencia (conferencia da implementacao propria em metricas_mot.py)
#   xmltodict  dependencia do motmetrics
# HOTA/CLEAR/Identity sao calculados por metricas_mot.py (reimplementacao do TrackEval, conferida contra o
# motmetrics em 31_conferir_metricas.py); o TrackEval original nao foi executado.
set -e
BASE=~/cerise2026/drones
mkdir -p $BASE/pylib_mot $BASE/validacao_videos
sudo docker run --rm -u $(id -u):$(id -g) -e HOME=/w/home -v $BASE:/w cerise26_drones:latest \
  bash -c "nice -n 19 pip install --no-cache-dir --no-deps --target /w/pylib_mot lap motmetrics xmltodict && python -c 'import sys; sys.path.insert(0,\"/w/pylib_mot\"); import lap, motmetrics, ultralytics, numpy, scipy, cv2; print(\"lap\", lap.__version__, \"motmetrics\", motmetrics.__version__, \"ultralytics\", ultralytics.__version__, \"numpy\", numpy.__version__, \"scipy\", scipy.__version__, \"cv2\", cv2.__version__)'"
echo FIM_LIBS
