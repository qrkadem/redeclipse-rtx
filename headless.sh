#!/bin/sh
# Runs Red Eclipse inside gamescope's headless backend: GPU rendered, no window on the desktop, no audio, separate home.
# usage: [W=width H=height BIN=binary] ./headless.sh <home dir> <seconds> <-x script and other game args...>
home="$1"; secs="$2"; shift 2
W="${W:-1600}"; H="${H:-900}"; BIN="${BIN:-./bin/amd64/redeclipse_linux}"
cd "$(dirname "$0")" || exit 1
# settings a test changes persist in the home config, so each run starts from the defaults
mkdir -p "$home" && rm -f "$home/config.cfg"
env -u WAYLAND_DISPLAY -u DISPLAY -u HYPRLAND_INSTANCE_SIGNATURE ALSOFT_DRIVERS=null SDL_AUDIODRIVER=dummy \
    timeout -k 5 "$secs" gamescope --backend headless -W "$W" -H "$H" -w "$W" -h "$H" -- \
    env SDL_VIDEODRIVER=x11 "$BIN" -h"$home" -dw"$W" -dh"$H" -df0 "$@"
status=$?
# killing gamescope at the time limit can leave the game running on its own, so end whatever ran from this home
pkill -KILL -f -- "-h$home " 2>/dev/null
exit $status
