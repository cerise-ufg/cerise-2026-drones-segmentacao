#!/bin/bash
cd ~/cerise2026/drones/sim
bash scripts/rodar_contagem_spark.sh figs scripts/figuras_sim.py --cenario cenario/reduzido --execucao saidas/run01 \
  --contagens 40_m:resultados/contagem_run01.json 30_m:resultados/contagem_run02a30.json \
  --varreduras 40_m:resultados/varredura_run01.json 30_m:resultados/varredura_run02a30.json \
  --vigia saidas/run01.vigia.log --saida evidencias --prefixo sim_ --modo botsort_cal >/dev/null
bash scripts/rodar_contagem_spark.sh amo_run01 scripts/figura_amostras.py --execucao saidas/run01 \
  --deteccoes resultados/contagem_run01_deteccoes_botsort_cal.csv --saida evidencias --prefixo sim_ >/dev/null
while sudo docker ps --format '{{.Names}}' | grep -qE 'cerise26_drones_(figs|amo_run01)'; do sleep 5; done
tail -n 3 logs/figs.log logs/amo_run01.log
ls -la evidencias/ | tail -12
