#!/usr/bin/env python3
"""Where does the SSD go idle inside a token?

The engine achieves ~2.73 GB/s against a device ceiling measured at 3.48 GB/s.
That gap is either (a) the drive sitting idle -- a scheduling problem worth
fixing -- or (b) the drive running slowly while busy, i.e. starved queue depth.
ExpertPool already tracks both (busy_s, depth_sum/depth_n); nothing printed it.

Prefill is excluded: stats are snapshotted after prefill and differenced.
"""
import os
import sys
import time

import mlx.core as mx

sys.path.insert(0, "/Users/agastya/Desktop/moe-stream")
from mlx_lm.models.cache import make_prompt_cache

import engine_120b as E

CEILING = 3.48          # GB/s, measured by fd_bench.py on this drive


def snap(pool):
    s = pool.stats()
    return dict(s, wall=time.perf_counter())


def main():
    n = int(os.environ.get("TOKENS", "96"))
    model, tok, pool = E.load_engine()
    ids = tok.encode(os.environ.get(
        "PROMPT", "Explain what a mixture-of-experts model is."))
    c = make_prompt_cache(model)
    y = mx.argmax(model(mx.array(ids)[None], cache=c)[:, -1], axis=-1)
    mx.eval(y)

    a = snap(pool)                       # post-prefill baseline
    times = []
    for _ in range(n):
        t = time.perf_counter()
        y = mx.argmax(model(y[None], cache=c)[:, -1], axis=-1)
        mx.eval(y)
        times.append(time.perf_counter() - t)
    b = snap(pool)

    wall = b["wall"] - a["wall"]
    gb = b["gb"] - a["gb"]
    busy = b["busy_s"] - a["busy_s"]
    blocked = b["blocked"] - a["blocked"]
    exposed = b["exposed_gb"] - a["exposed_gb"]
    acc = b["total"] - a["total"]
    reads = (b["miss"] - a["miss"]) + (b["issued"] - a["issued"])

    print(f"\n=== decode only, {n} tokens, {pool.n_slots} slots ===")
    print(f"  wall              {wall:8.2f} s   ({wall/n*1e3:.0f} ms/token, "
          f"{n/wall:.2f} tok/s)")
    print(f"  bytes read        {gb*1024/n:8.0f} MB/token   ({gb:.2f} GiB total)")
    print()
    print(f"  device busy       {busy:8.2f} s   {100*busy/wall:5.1f}% duty cycle")
    print(f"  device IDLE       {wall-busy:8.2f} s   {100*(1-busy/wall):5.1f}%"
          f"  <- the schedulable gap")
    print(f"  rate while busy   {gb*1.0737/busy:8.2f} GB/s  "
          f"({100*gb*1.0737/busy/CEILING:.0f}% of the {CEILING} GB/s ceiling)")
    print(f"  rate overall      {gb*1.0737/wall:8.2f} GB/s")
    print(f"  mean queue depth  {b['depth']:8.2f}   (device peaks at 2-4)")
    print()
    print(f"  acquire blocked   {blocked:8.2f} s   {100*blocked/wall:5.1f}% of wall")
    print(f"  exposed bytes     {exposed*1024/n:8.0f} MB/token  "
          f"(demand misses, nothing hiding them)")
    print(f"  prefetch bails    {b['bail']-a['bail']:8d}")
    print(f"  reads issued      {reads:8d}  ({reads/n:.1f}/token of "
          f"{acc/n:.0f} accesses)")
    print()
    ideal = gb * 1.0737 / CEILING
    print(f"  if the drive never idled: {ideal/n*1e3:.0f} ms/token -> "
          f"{n/ideal:.2f} tok/s  ({wall/ideal:.2f}x)")


if __name__ == "__main__":
    main()
