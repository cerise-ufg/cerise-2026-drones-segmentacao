#!/bin/bash
# Monta os MP4 H.264 (<= 20 MB) a partir dos quadros renderizados em video_tmp/
B=~/cerise2026/drones/validacao_videos
mkdir -p $B/evidencias
for S in uav0000073_04464_v uav0000120_04775_v; do
  for CRF in 22 25 28 31; do
    nice -n 19 ffmpeg -y -loglevel error -framerate 25 -i $B/video_tmp/$S/%06d.jpg -c:v libx264 -preset slow -crf $CRF \
      -pix_fmt yuv420p -movflags +faststart -threads 4 $B/evidencias/v_video_$S.mp4
    SZ=$(stat -c %s $B/evidencias/v_video_$S.mp4)
    echo "$S crf=$CRF tamanho=$SZ"
    [ $SZ -le 19500000 ] && break
  done
  ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames,r_frame_rate -show_entries format=duration,size -of compact $B/evidencias/v_video_$S.mp4
done
