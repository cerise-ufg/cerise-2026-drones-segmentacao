#!/bin/bash
echo "== journal (14:00-14:06): sessao grafica / firefox / gnome-shell"
sudo journalctl --since "2026-10-26 13:55" --until "2026-10-26 14:08" --no-pager 2>/dev/null | grep -iE "firefox|gnome-shell|session-|gdm|Xorg|kiosk|oom|Removed session|New session|logged out" | cut -c1-200 | tail -40
echo "== firefox antes? (journal desde 09:00)"
sudo journalctl --since "2026-10-26 09:00" --until "2026-10-26 13:55" --no-pager 2>/dev/null | grep -iE "firefox" | tail -5 | cut -c1-200
echo "== autologin"
grep -iE "autologin|AutomaticLogin" /etc/gdm3/custom.conf 2>/dev/null
cat ~/.config/autostart/energia-ia-kiosk.desktop
