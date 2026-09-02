#!/usr/bin/env python3
"""Is the engine's 3.05 GB/s (vs a 3.50 saturated ceiling) burstiness?

gil_io.py excluded the two standing hypotheses: a spinning MLX main thread costs
nothing at any thread count (GIL is not it), and mean queue depth 2.79 is not it
either -- a *steady* depth of 2 already reaches 3.43.

What is left is the shape of the queue. The engine does not keep a queue; per
layer it submits ~2.1 blob reads and then blocks on all of them, so every burst
ends at depth 1 and the device drains between layers. Mean depth over busy time
can read 2.79 while most of the bytes move at depth 1-2.

This measures exactly that: BURST reads submitted, joined, repeat -- against the
same total bytes through a saturated queue. Same real pool, same real segments,
same drive, fresh random offsets per run.

BURST=0 means the saturated control.
"""
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/Users/agastya/Desktop/moe-stream")

NSLOTS = int(os.environ.get("SLOTS", "64"))
NREAD = int(os.environ.get("NREAD", "256"))
NTHREAD = int(os.environ.get("THREADS", "4"))
BURST = int(os.environ.get("BURST", "0"))

os.environ["MOE_COLD"] = "1"
os.environ["PF_SLOTS"] = str(NSLOTS)
os.environ["MOE_STAGE"] = "1"

import engine_120b as E

DIR = os.path.expanduser("~/Desktop/moe-stream/model-120b")

if __name__ == "__main__":
    pool = E.ExpertPool(DIR, n_slots=NSLOTS, n_workers=NTHREAD)
    rng = random.Random(int(os.environ.get("SEED", "0")))
    jobs = [(rng.randrange(pool.n_layers), rng.randrange(pool.n_experts))
            for _ in range(NREAD)]
    ex = ThreadPoolExecutor(NTHREAD)

    b0, d0, n0 = pool.bytes_read, pool.depth_sum, pool.depth_n
    busy0 = pool.busy_s
    t0 = time.perf_counter()
    if BURST == 0:
        # saturated: every read queued at once, workers never starve
        futs = [ex.submit(pool._read, i % NSLOTS, l, e)
                for i, (l, e) in enumerate(jobs)]
        for f in futs:
            f.result()
    else:
        # the engine's shape: submit a layer's worth, block on all of it, repeat
        for s in range(0, len(jobs), BURST):
            chunk = jobs[s:s + BURST]
            futs = [ex.submit(pool._read, (s + i) % NSLOTS, l, e)
                    for i, (l, e) in enumerate(chunk)]
            for f in futs:
                f.result()
    dt = time.perf_counter() - t0
    gb = (pool.bytes_read - b0) / 1e9
    busy = pool.busy_s - busy0
    depth = (pool.depth_sum - d0) / max(1, pool.depth_n - n0)
    print(f"BURST={BURST:<3} THREADS={NTHREAD}  {gb/dt:6.3f} GB/s overall   "
          f"{gb/busy:6.3f} GB/s while busy   depth {depth:4.2f}   "
          f"duty {100*busy/dt:5.1f}%   ({dt:.2f}s)")
