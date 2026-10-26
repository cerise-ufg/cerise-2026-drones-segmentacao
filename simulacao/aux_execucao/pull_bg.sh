#!/bin/bash
cd ~/cerise2026/drones/sim/tmp
echo "inicio $(date -Is)" > pull.log
sudo nice -n 19 docker pull --platform linux/arm64 nvcr.io/nvidia/isaac-sim:5.1.0 >> pull.log 2>&1
echo "rc=$? fim $(date -Is)" >> pull.log
