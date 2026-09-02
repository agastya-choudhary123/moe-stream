#!/bin/zsh
cd /Users/agastya/Desktop/moe-stream
export MOE_COLD=1 PF_SLOTS=600 PF_DEPTH=1 TOKENS=96 MOE_EVICT=lfu
for round in 1 2; do
  if (( round % 2 == 1 )); then order=(4 8 12); else order=(12 8 4); fi
  for w in $order; do
    echo "=== round $round  PF_WORKERS=$w ==="
    PF_WORKERS=$w python3 engine_120b.py 2>&1 | grep -E "tok/s|hit:|read "
  done
done
