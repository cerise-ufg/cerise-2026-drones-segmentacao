#!/bin/bash
echo "== processos (somente leitura)"
ps -eo pid,user,etimes,comm,args --sort=etimes | grep -E "firefox|kiosk|gnome-shell|Xwayland|Xorg|gdm|pipewire|dbus-daemon" | grep -v grep | cut -c1-170 | head -30
echo "== dmesg OOM (processos mortos)"
sudo dmesg -T | grep -E "Killed process|oom-kill:" | tail -20 | cut -c1-230
echo "== loginctl"
loginctl list-sessions 2>/dev/null | head
echo "== kiosk.sh (cabecalho, somente leitura)"
head -40 ~/energia-ia/kiosk.sh 2>/dev/null
systemctl --user -M energiainteligente@ list-units 2>/dev/null | grep -i kiosk | head
ls -la ~/.config/autostart 2>/dev/null
