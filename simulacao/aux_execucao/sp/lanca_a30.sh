#!/bin/bash
cd ~/cerise2026/drones/sim
ls cenario/reduzido_a30/
sudo docker logs cerise26_isaac_run01 > logs/isaac_run01.log 2>&1
for n in smoke3 smoke4 smoke6; do sudo docker rm cerise26_isaac_$n >/dev/null 2>&1; done
ACCEPT_EULA=Y PRIVACY_CONSENT=N ATIVOS_LOCAIS=/home/energiainteligente/cerise2026/drones/sim/isaac_ativos \
  bash isaac/rodar_isaac.sh run02a30 --cenario /cenario/reduzido_a30 --fps 5 --sem_visao_geral 2>&1 | tail -2
bash scripts/rodar_contagem_spark.sh cont_run01 scripts/contar.py --execucao saidas/run01 --cenario cenario/reduzido \
  --detector yolo --pesos /w/resultados/best_yolo11s_visdrone_pessoa_960.pt --imgsz 1280 \
  --rastreadores botsort,chao,botsort_pose,botsort_semgmc --saida resultados/contagem_run01.json --figura
sudo du -sh saidas/run01
for c in D1 D2 D3 D4 visao_geral; do echo $c $(ls saidas/run01/$c/rgb | wc -l) $(wc -l < saidas/run01/$c/poses.csv) $(wc -l < saidas/run01/$c/caixas_gt.csv 2>/dev/null); done
python3 -c "import json;m=json.load(open('saidas/run01/meta.json'));print({k:m[k] for k in ('modo_pessoas','quadros','s_por_quadro','s_render_por_quadro','tempo_laco_s','avisos','fonte_gabarito')})"
