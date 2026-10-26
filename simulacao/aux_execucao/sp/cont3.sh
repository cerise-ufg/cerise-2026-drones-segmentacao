#!/bin/bash
cd ~/cerise2026/drones/sim
while sudo docker ps --format '{{.Names}}' | grep -qx cerise26_isaac_run02a30; do sleep 10; done
sudo docker logs cerise26_isaac_run02a30 > logs/isaac_run02a30.log 2>&1
grep -E 'FIM' logs/isaac_run02a30.log; tail -1 saidas/run02a30.vigia.log
for c in D1 D2 D3 D4; do echo $c $(ls saidas/run02a30/$c/rgb | wc -l) $(wc -l < saidas/run02a30/$c/caixas_gt.csv); done
bash scripts/rodar_contagem_spark.sh cont_run02a30 scripts/contar.py --execucao saidas/run02a30 --cenario cenario/reduzido_a30 \
  --detector yolo --pesos /w/resultados/best_yolo11s_visdrone_pessoa_960.pt --imgsz 1280 \
  --rastreadores botsort,botsort_cal,chao,botsort_pose,botsort_semgmc --saida resultados/contagem_run02a30.json --figura >/dev/null
while sudo docker ps --format '{{.Names}}' | grep -qx cerise26_drones_cont_run02a30; do sleep 5; done
grep -E '^\[' logs/cont_run02a30.log
bash scripts/rodar_contagem_spark.sh cont_run01 scripts/contar.py --execucao saidas/run01 --cenario cenario/reduzido \
  --detector yolo --pesos /w/resultados/best_yolo11s_visdrone_pessoa_960.pt --imgsz 1280 \
  --rastreadores botsort,botsort_cal,chao,botsort_pose,botsort_semgmc --saida resultados/contagem_run01.json --figura >/dev/null
bash scripts/rodar_contagem_spark.sh varr_a30 scripts/varredura_botsort.py --execucao saidas/run02a30 --cenario cenario/reduzido_a30 --cache resultados/contagem_run02a30_cache_det.npz --saida resultados/varredura_run02a30.json >/dev/null
bash scripts/rodar_contagem_spark.sh video_run01 scripts/video_relatorio.py --execucao saidas/run01 --contagem resultados/contagem_run01.json \
  --deteccoes resultados/contagem_run01_deteccoes_botsort_cal.csv --quadros_dir tmp/video_run01 >/dev/null
while sudo docker ps --format '{{.Names}}' | grep -qE 'cerise26_drones_(cont|varr|video)_'; do sleep 5; done
python3 tmp/sp/resumo_cont.py resultados/contagem_run02a30.json | grep -v '^tau 0.35'
bash scripts/montar_video.sh tmp/video_run01 evidencias/sim_video_run01.mp4 15
