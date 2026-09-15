#!/usr/bin/env bash
# render_chunk.sh <first> <last> <out.mp4> [W H]
# Renders a frame range straight to h264. Chunks are encoded with closed GOPs so
# they can be concatenated without re-encoding (see tools/render_video.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIRST=$1; LAST=$2; OUT=$3; W="${4:-1080}"; H="${5:-1920}"
TEXLIST=$(mktemp); printf '%s\n%s\n' "$ROOT/assets/text0.raw" "$ROOT/assets/text1.raw" > "$TEXLIST"
"$ROOT/src/render" "$ROOT/analysis/timeline.csv" "$W" "$H" "$FIRST" "$LAST" "$TEXLIST" 2>/dev/null \
 | ffmpeg -hide_banner -loglevel error -y \
     -f rawvideo -pixel_format rgb24 -video_size "${W}x${H}" -framerate 30 -i - \
     -c:v libx264 -preset slow -crf 23 -pix_fmt yuv420p -profile:v high -level 4.2 \
     -x264-params "keyint=60:min-keyint=12:scenecut=40:open-gop=0" \
     -an "$OUT"
rm -f "$TEXLIST"
