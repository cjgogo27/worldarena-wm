#!/usr/bin/env bash
# ============================================
# extract_videos.sh
# Copy all showcase videos to the github-pages-site/videos/ directory
# ============================================
set -euo pipefail

DST="/data/alice/cjtest/github-pages-site/videos"
mkdir -p "$DST"

echo "Extracting videos for GitHub Pages site..."

# --- Case 10: Three-way comparison ---
cp /data/alice/cjtest/model_repros/worldarena_abot_public/outputs/test10_t1_flat/episode10.mp4 "$DST/episode10_abot.mp4"
echo "  ✅ episode10_abot.mp4"

cp /data/alice/cjtest/model_repros/worldarena_wan_public/outputs/test10_t1_flat/episode10.mp4 "$DST/episode10_wan.mp4"
echo "  ✅ episode10_wan.mp4"

cp /data/alice/cjtest/VideoX-Fun/eval_ckpt300_test10/episode10.mp4 "$DST/episode10_sft.mp4"
echo "  ✅ episode10_sft.mp4"

# --- Case 106: Three-way comparison ---
cp /data/alice/cjtest/model_repros/worldarena_abot_public/outputs/test10_t1_flat/episode106.mp4 "$DST/episode106_abot.mp4"
echo "  ✅ episode106_abot.mp4"

cp /data/alice/cjtest/model_repros/worldarena_wan_public/outputs/test10_t1_flat/episode106.mp4 "$DST/episode106_wan.mp4"
echo "  ✅ episode106_wan.mp4"

cp /data/alice/cjtest/VideoX-Fun/eval_ckpt300_test10/episode106.mp4 "$DST/episode106_sft.mp4"
echo "  ✅ episode106_sft.mp4"

# --- Failure cases ---
cp /data/alice/cjtest/model_repros/worldarena_wan_public/outputs/test10_t1_flat/episode1.mp4 "$DST/episode1_wan_fail.mp4"
echo "  ✅ episode1_wan_fail.mp4"

cp /data/alice/cjtest/model_repros/worldarena_abot_public/outputs/test10_t1_flat/episode100.mp4 "$DST/episode100_abot_fail.mp4"
echo "  ✅ episode100_abot_fail.mp4"

cp /data/alice/cjtest/model_repros/worldarena_wan_public/outputs/test10_t1_flat/episode10.mp4 "$DST/episode10_wan_fail.mp4"
echo "  ✅ episode10_wan_fail.mp4"

# --- SeedVR before/after (Wan2.1 episode105) ---
cp /data/alice/cjtest/model_repros/worldarena_wan_public/outputs/test10_t1_flat/episode105.mp4 "$DST/episode105_wan_raw.mp4"
echo "  ✅ episode105_wan_raw.mp4"

cp /data/alice/cjtest/model_repros/worldarena_wan_public/outputs/test10_t1_seedvr/episode105.mp4 "$DST/episode105_wan_seedvr.mp4"
echo "  ✅ episode105_wan_seedvr.mp4"

echo ""
echo "Done! $(ls "$DST"/*.mp4 | wc -l) video files copied to $DST"
ls -lh "$DST"/
