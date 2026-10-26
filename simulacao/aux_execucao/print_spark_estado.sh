#!/bin/bash
echo '$ free -g; uptime'
free -g | head -2; uptime
echo '$ sudo docker ps --format "{{.Names}}  {{.Status}}"'
sudo docker ps --format '{{.Names}}  {{.Status}}'
echo '$ df -h / | tail -1'
df -h / | tail -1
