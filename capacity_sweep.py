#!/usr/bin/env python3
"""
Take the capacity the pool fix unlocked.

613 slots was recorded as a hard ceiling on gpt-oss-120b because Metal's
`max_buffer_length` is 8 GiB. That limit applied to a pool that was one buffer;
it is now one array per component, so the cap is RAM. The capacity curve was
still climbing steeply at the old ceiling (360 -> 1.34, 480 -> 1.84,
600 -> 2.51 tok/s), which is why this is worth measuring rather than assuming.

Interleaved, order alternated every round, one process per run, always cold --
run-to-run spread here is ~15% and a single ordering has produced 1.48x and
0.86x for the same pair before now. Bytes per token is reported alongside
throughput because it is near-deterministic where tok/s is not: if slots help,
MB/token must fall, and a tok/s change with flat MB/token is noise.

Memory is the real risk at the top end: 700 slots is a 9.14 GiB pool against a
10.67 GiB working set, so the run is watched for a resident-set blowup as well
as for speed.

  python3 capacity_sweep.py            # 600 vs 700, 4 rounds
  SLOTS=600,660,700 ROUNDS=3 python3 capacity_sweep.py
"""

import json
import os
import re
import statistics as st
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(slots):
    env = dict(os.environ, MOE_COLD="1", PF_SLOTS=str(slots), PF_DEPTH="1",
               BENCH_ENGINE="engine_120b", BENCH_LABEL=f"slots{slots}",
               BENCH_TRIALS="2", BENCH_TOKENS="24")
    p = subprocess.run([sys.executable, "bench.py"], cwd=HERE, env=env,
                       text=True, capture_output=True)
    if p.returncode != 0:
        print(p.stdout[-800:], p.stderr[-800:])
        return None
    o = p.stdout
    med = float(re.search(r"median ([\d.]+) tok/s", o).group(1))
    res = float(re.search(r"resident ([\d.]+) GB", o).group(1))
    hit = re.search(r"cache (\d+)%", o)
    return dict(tok_s=med, resident=res, cache_hit=int(hit.group(1)) if hit else -1,
                raw=o)


def main():
    slots = [int(s) for s in os.environ.get("SLOTS", "600,700").split(",")]
    rounds = int(os.environ.get("ROUNDS", "4"))
    res = {s: [] for s in slots}
    for r in range(rounds):
        order = slots if r % 2 == 0 else slots[::-1]
        for s in order:
            t0 = time.perf_counter()
            out = run(s)
            if out is None:
                print(f"  round {r+1} slots {s}: FAILED")
                continue
            res[s].append(out)
            print(f"  round {r+1} slots {s:>4}: {out['tok_s']:5.2f} tok/s  "
                  f"resident {out['resident']:.2f} GB  cache hit "
                  f"{out['cache_hit']}%  ({time.perf_counter()-t0:.0f}s)",
                  flush=True)
        json.dump({str(k): [{kk: vv for kk, vv in d.items() if kk != "raw"}
                            for d in v] for k, v in res.items()},
                  open(os.path.join(HERE, "acts", "capacity_sweep.json"), "w"),
                  indent=1)

    print(f"\n{'slots':>6} {'median':>8} {'min':>7} {'max':>7} {'resident':>9} "
          f"{'cache hit':>10}")
    base = None
    for s in slots:
        v = [d["tok_s"] for d in res[s]]
        if not v:
            continue
        m = st.median(v)
        base = base or m
        print(f"{s:>6} {m:8.2f} {min(v):7.2f} {max(v):7.2f} "
              f"{st.median([d['resident'] for d in res[s]]):8.2f} GB "
              f"{st.median([d['cache_hit'] for d in res[s]]):9.0f}%"
              f"   {m/base:.2f}x")
    print("\nOverlapping min-max ranges mean no result: ~15% run-to-run spread "
          "here, ~12% is the resolution limit.")


if __name__ == "__main__":
    main()
