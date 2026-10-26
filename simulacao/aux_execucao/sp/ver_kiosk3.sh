#!/bin/bash
last -x 2>/dev/null | grep -vE "pts/" | head -12
echo "== gdm/autologin/gnome-shell do usuario desde o boot"
sudo journalctl -b --no-pager 2>/dev/null | grep -E "gdm-autologin|gdm-password|pam_unix\(gdm|gnome-shell\[|firefox\[|kiosk" | grep -v "gdm\]" | head -15 | cut -c1-200
echo "== gnome-shell do usuario 1000 agora?"; pgrep -u 1000 -a gnome-shell | head -2; pgrep -u 1000 -a firefox | head -2
echo "== journal 14:03:40-14:04:30 completo (filtrado)"
sudo journalctl --since "2026-10-26 14:03:40" --until "2026-10-26 14:04:30" --no-pager 2>/dev/null | grep -vE "kernel: \[|kernel:  " | cut -c1-180 | head -40
