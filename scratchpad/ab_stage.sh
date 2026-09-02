#!/bin/zsh
# A/B the staging-buffer read path. Uses profile_io so the primary metric is
# "rate while busy" -- far less noisy than tok/s, which has a ~12% floor here.
cd /Users/agastya/Desktop/moe-stream
export MOE_COLD=1 PF_SLOTS=600 PF_DEPTH=1 TOKENS=96 MOE_EVICT=lfu
P=/private/tmp/claude-501/-Users-agastya-Desktop-moe-stream/32819ea4-feb4-45b0-94d1-5314981e9000/scratchpad/profile_io.py
for round in 1 2; do
  if (( round % 2 == 1 )); then order=(0 1); else order=(1 0); fi
  for s in $order; do
    echo "=== round $round  MOE_STAGE=$s ==="
    MOE_STAGE=$s python3 -u $P 2>&1 \
      | grep -E "tok/s|rate while busy|duty cycle|queue depth|MB/token"
  done
done
