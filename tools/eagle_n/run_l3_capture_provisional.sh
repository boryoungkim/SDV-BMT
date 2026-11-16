#!/usr/bin/env bash
# PROVISIONAL L3 (user decision 2026-11-15): same as run_l3_capture.sh with the fixed_rate 0.8 Hz capture plan, because
# the original SDV-BMT asap measure watchdog (154 s) kills the 500-item Eagle-N window. See the plan header.
PLAN=configs/experiments/deeplabv3plus-r18-eagle-n-capture-provisional-0p8hz.yaml exec "$(dirname "$0")/run_l3_capture.sh"
