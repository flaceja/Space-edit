#!/usr/bin/env bash
# render_video.sh <out.mp4> [W H] - renders the whole timeline and muxes the track
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/out/space-edit.mp4}"
W="${2:-1080}"; H="${3:-1920}"
FPS=30
AUDIO="${AUDIO:-$ROOT/audio/singularity.mp3}"
TL="$ROOT/analysis/timeline.csv"
N=$(grep -vc '^#' "$TL")
mkdir -p "$(dirname "$OUT")" "$ROOT/assets"
TEXLIST=$(mktemp)
printf '%s\n%s\n' "$ROOT/assets/text0.raw" "$ROOT/assets/text1.raw" > "$TEXLIST"
echo "rendering $N frames at ${W}x${H} -> $OUT"
"$ROOT/src/render" "$TL" "$W" "$H" 0 $((N-1)) "$TEXLIST" \
 | ffmpeg -hide_banner -loglevel warning -y \
     -f rawvideo -pixel_format rgb24 -video_size "${W}x${H}" -framerate $FPS -i - \
     -i "$AUDIO" \
     -map 0:v:0 -map 1:a:0 \
     -c:v libx264 -preset slow -crf 23 -pix_fmt yuv420p -profile:v high -level 4.2 \
     -x264-params "keyint=60:min-keyint=30:scenecut=0" \
     -c:a aac -b:a 256k -ar 44100 \
     -movflags +faststart -shortest "$OUT"
rm -f "$TEXLIST"
ls -la "$OUT"
