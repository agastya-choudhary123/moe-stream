#!/bin/zsh
# Interleaved A/B of the eviction policy on the real 120b engine.
# Order alternates each round so drift cannot favour one arm.
cd /Users/agastya/Desktop/moe-stream
export MOE_COLD=1 PF_SLOTS=600 PF_DEPTH=1 TOKENS=32
for round in 1 2 3; do
  if (( round % 2 == 1 )); then order=(lru lfu); else order=(lfu lru); fi
  for pol in $order; do
    echo "=== round $round  MOE_EVICT=$pol ==="
    MOE_EVICT=$pol python3 engine_120b.py 2>&1 \
      | grep -E "tok/s|hit:|read |slots x|predict recall"
  done
done
