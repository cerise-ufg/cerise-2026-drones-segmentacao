#!/bin/bash
cd ~/cerise2026/drones/sim
bash scripts/rodar_contagem_spark.sh cont_run01 scripts/contar.py --execucao saidas/run01 --cenario cenario/reduzido \
  --detector yolo --pesos /w/resultados/best_yolo11s_visdrone_pessoa_960.pt --imgsz 1280 \
  --rastreadores botsort,botsort_cal,chao,botsort_pose,botsort_semgmc --saida resultados/contagem_run01.json --figura >/dev/null
while sudo docker ps --format '{{.Names}}' | grep -qx cerise26_drones_cont_run01; do sleep 5; done
grep -E '^\[|cache' logs/cont_run01.log
bash scripts/rodar_contagem_spark.sh video_run01 scripts/video_relatorio.py --execucao saidas/run01 --contagem resultados/contagem_run01.json \
  --deteccoes resultados/contagem_run01_deteccoes_botsort_cal.csv --quadros_dir tmp/video_run01 >/dev/null
bash scripts/rodar_contagem_spark.sh amo_run01 scripts/figura_amostras.py --execucao saidas/run01 \
  --deteccoes resultados/contagem_run01_deteccoes_botsort_cal.csv --saida evidencias --prefixo sim_ >/dev/null
while sudo docker ps --format '{{.Names}}' | grep -qE 'cerise26_drones_(video|amo)_run01'; do sleep 5; done
tail -n 2 logs/video_run01.log logs/amo_run01.log 2>&1 | grep -v '^$'
bash scripts/montar_video.sh tmp/video_run01 evidencias/sim_video_run01.mp4 15
bash tmp/sp/espera.sh cerise26_isaac_run02a30 xx 2 2
