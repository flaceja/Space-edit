#!/usr/bin/env bash
# preview.sh <timeline.csv> <outdir> [W H] [texlist]  -> one PNG per timeline row
set -e
TL="$1"; OUT="$2"; W="${3:-1080}"; H="${4:-1920}"; TEX="${5:-}"
mkdir -p "$OUT"; rm -f "$OUT"/f*.png
N=$(grep -vc '^#' "$TL")
/home/user/Space-edit/src/render "$TL" "$W" "$H" 0 $((N-1)) $TEX 2>/dev/null \
  | ffmpeg -hide_banner -v error -y -f rawvideo -pix_fmt rgb24 -s "${W}x${H}" -i - \
      -vf scale=540:-1 "$OUT/f%02d.png"
ls "$OUT"
