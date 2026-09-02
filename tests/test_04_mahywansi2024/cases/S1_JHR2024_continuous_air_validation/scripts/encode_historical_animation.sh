#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:?usage: encode_historical_animation.sh OUTPUT_DIR}"
frames="${output_dir}/frames/frame_%04d.png"
mp4="${output_dir}/historical_stage1_coarse_0_to_5p22.mp4"
gif="${output_dir}/historical_stage1_coarse_0_to_5p22.gif"

nice -n 19 ffmpeg -hide_banner -loglevel error -y \
    -framerate 10 -i "$frames" \
    -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -threads 1 \
    "$mp4"

nice -n 19 ffmpeg -hide_banner -loglevel error -y \
    -framerate 10 -i "$frames" \
    -vf 'fps=10,scale=1000:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=192[p];[s1][p]paletteuse=dither=sierra2_4a' \
    -loop 0 -threads 1 \
    "$gif"

ffprobe -v error \
    -show_entries format=duration,size \
    -show_entries stream=width,height,avg_frame_rate,nb_frames \
    -of json "$mp4"
