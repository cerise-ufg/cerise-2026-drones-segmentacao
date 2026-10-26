#!/bin/bash
ls -la ~/braco/dataset | head -12
for d in ~/braco/dataset/*/; do echo "$d $(ls "$d" | wc -l)"; done 2>/dev/null | head -12
cat ~/braco/continuar.sh 2>/dev/null | head -40
ps aux | grep -c "[g]z sim\|[i]gn gazebo"
