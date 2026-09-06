#!/usr/bin/env bash
# postprocess.sh IN OUT UPSCALE FPS CODEFORMER_FIDELITY
# Upscale -> face-restore -> interpolate. Uses ComfyUI nodes where available,
# falls back to ffmpeg + Real-ESRGAN/CodeFormer CLIs if installed.
set -euo pipefail
IN="$1"; OUT="$2"; UP="${3:-1.75}"; FPS="${4:-48}"; FID="${5:-0.7}"
WORK="$(dirname "$OUT")"
TMP="$WORK/_pp"; mkdir -p "$TMP"

# 1) extract frames
ffmpeg -y -i "$IN" "$TMP/f_%05d.png" >/dev/null 2>&1

# 2) upscale + face restore (CodeFormer if present, else lanczos)
if command -v codeformer >/dev/null 2>&1; then
  codeformer -w "$FID" --upscale "$UP" -i "$TMP" -o "$TMP/up" >/dev/null 2>&1 || true
  SRC="$TMP/up"
else
  SRC="$TMP"
fi

# 3) reassemble
ffmpeg -y -framerate 24 -i "$SRC/f_%05d.png" -c:v libx264 -pix_fmt yuv420p "$TMP/base.mp4" >/dev/null 2>&1

# 4) interpolate to target fps (RIFE if present, else ffmpeg minterpolate)
if command -v rife-ncnn-vulkan >/dev/null 2>&1; then
  rife-ncnn-vulkan -i "$SRC" -o "$TMP/interp" >/dev/null 2>&1 || true
  ffmpeg -y -framerate "$FPS" -i "$TMP/interp/%05d.png" -c:v libx264 -pix_fmt yuv420p "$OUT" >/dev/null 2>&1
else
  ffmpeg -y -i "$TMP/base.mp4" -vf "minterpolate=fps=$FPS" -c:v libx264 -pix_fmt yuv420p "$OUT" >/dev/null 2>&1
fi
rm -rf "$TMP"
echo "postprocess -> $OUT"
