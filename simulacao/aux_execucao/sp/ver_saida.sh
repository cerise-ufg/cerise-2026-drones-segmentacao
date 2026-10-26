#!/bin/bash
# uso: ver_saida.sh <nome>
N=$1
S=~/cerise2026/drones/sim/saidas/$N
echo "== conteiner"; sudo docker ps -a --format '{{.Names}} {{.Status}}' | grep cerise26_isaac_$N
echo "== arquivos"; sudo ls -la $S 2>/dev/null | head; for d in $S/*/; do echo "$d rgb=$(sudo ls $d/rgb 2>/dev/null | wc -l) cx=$(sudo wc -l < $d/caixas_gt.csv 2>/dev/null)"; done
echo "== meta"; sudo python3 -c "import json;m=json.load(open('$S/meta.json'));[print(k, m[k]) for k in ('modo_pessoas','quadros','s_por_quadro','s_render_por_quadro','avisos','tempo_laco_s')]" 2>/dev/null
echo "== caixas D1"; sudo head -5 $S/D1/caixas_gt.csv 2>/dev/null
echo "== gabarito"; sudo head -4 $S/gabarito_pessoas.csv 2>/dev/null
echo "== vigia"; tail -3 ~/cerise2026/drones/sim/saidas/$N.vigia.log; awk -F, 'NR>2 && $2+0>0 {if(min==""||$2<min)min=$2} END{print "min available MB:",min}' ~/cerise2026/drones/sim/saidas/$N.vigia.log
echo "== log app (cerise_sim + erros)"; sudo docker logs cerise26_isaac_$N 2>&1 | grep -E "cerise_sim|Error|ERROR|Traceback|error:|Exception" | grep -v "omni.kit.registry\|\[Warning\]" | tail -${2:-40}
