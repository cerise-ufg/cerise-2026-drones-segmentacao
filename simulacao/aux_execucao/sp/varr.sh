#!/bin/bash
cd ~/cerise2026/drones/sim
bash scripts/rodar_contagem_spark.sh varr_run01 scripts/varredura_botsort.py --execucao saidas/run01 --cenario cenario/reduzido --cache resultados/contagem_run01_cache_det.npz --saida resultados/varredura_run01.json >/dev/null
while sudo docker ps --format '{{.Names}}' | grep -qx cerise26_drones_varr_run01; do sleep 5; done
grep 'False' logs/varr_run01.log | cut -c1-330
python3 - <<'PY'
import json
d=json.load(open('resultados/contagem_run01.json'))
print({k:(round(v['ap'],3),round(v.get('revocacao_max',0),3),v['n_gt'],v['n_det']) for k,v in d['deteccao_bruta']['ap'].items()})
PY
bash tmp/sp/espera.sh cerise26_isaac_run02a30 xx 2 2
