#!/bin/bash
# texto do print de terminal do render do close-up (log do app + vigia + ffprobe do clipe)
cd ~/cerise2026/drones/sim
{
  echo "conteiner cerise26_isaac_closeup01 iniciado (--memory 24g, --cpus 8, --network none); vigia de memoria a cada 2 s"
  sudo docker logs cerise26_isaac_closeup01 2>&1 | grep -E '^\[closeup' | grep -v 'RTX:' | sed -E 's/; materiais biblioteca; fatores de UV .*$/; materiais da biblioteca NVIDIA/' \
    | awk '/quadro [0-9]+ /{n++; if (n % 2 == 0 && $0 !~ /HEROI/) next} {print}'
  echo "--- vigia de memoria (available do Spark, MB; memoria do conteiner) ---"
  sed -n '1p' saidas/closeup01.vigia.log
  awk -F, 'NR>2 && NF==4 {print}' saidas/closeup01.vigia.log | awk 'NR%12==1'
  tail -1 saidas/closeup01.vigia.log
  echo "--- clipe ---"
  ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames,r_frame_rate -show_entries format=duration,size -of compact evidencias/sim_closeup.mp4
} > closeup/print_render.txt
wc -l closeup/print_render.txt
