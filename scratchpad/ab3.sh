#!/bin/zsh
# 3-arm interleaved A/B. "asis" reproduces the shipped engine (freq never
# incremented => effectively random eviction), so the baseline is real.
# TOKENS raised to 96 so prefill bytes are not charged so heavily to each token.
cd /Users/agastya/Desktop/moe-stream
export MOE_COLD=1 PF_SLOTS=600 PF_DEPTH=1 TOKENS=96
for round in 1 2 3; do
  case $round in
    1) order=(asis lfu lru);;
    2) order=(lru asis lfu);;
    3) order=(lfu lru asis);;
  esac
  for pol in $order; do
    echo "=== round $round  MOE_EVICT=$pol ==="
    MOE_EVICT=$pol python3 engine_120b.py 2>&1 | grep -E "tok/s|hit:|read |predict recall"
  done
done
