#!/bin/bash
# Monta o MP4 H.264 (<= 25 MB) a partir dos quadros do video_relatorio.py (ffmpeg/libx264 do host, nice 19).
# Uso: bash scripts/montar_video.sh tmp/video_run01 evidencias/sim_video_run01.mp4 [quadros_por_segundo=15]
set -e
SRC=$1; OUT=$2; FPS=${3:-15}
mkdir -p "$(dirname "$OUT")"
for CRF in 23 26 29 32 35; do
  nice -n 19 ffmpeg -y -loglevel error -framerate "$FPS" -i "$SRC/%06d.jpg" -c:v libx264 -preset slow -crf $CRF \
    -pix_fmt yuv420p -movflags +faststart -threads 4 "$OUT"
  SZ=$(stat -c %s "$OUT")
  echo "crf=$CRF tamanho=$SZ"
  [ "$SZ" -le 24500000 ] && break
done
ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames,r_frame_rate -show_entries format=duration,size -of compact "$OUT"
