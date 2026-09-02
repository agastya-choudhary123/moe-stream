#!/usr/bin/env python3
"""
One command that reproduces every headline number in the README.

    python3 repro.py                 # both models, full run (~10 min)
    python3 repro.py --quick         # 1 trial each  (~4 min)
    python3 repro.py --model 120b    # just gpt-oss-120b

Nothing is downloaded. It runs against the repacked stores already on disk and
fails loudly if they are missing rather than quietly measuring something else.

Two things it does that a plain benchmark would not:

  It verifies the pool before timing it. The pool spent this project's whole
  history silently returning the wrong scales for roughly half its slots (see
  HANDOFF.md), which cost nothing in throughput and everything in output
  quality, so "it ran fast" is not evidence that it ran correctly.

  It forces MOE_COLD=1. experts.bin is larger than RAM, so whatever the page
  cache happens to hold hands out free hits, and identical code has measured
  anywhere from 2 to 8 tok/s depending on what else was resident. Cold reads
  make the number depend only on the SSD and the slot pool.

Run-to-run spread on this machine is ~15%, so a single median means little and
the spread is printed with it. Differences under ~12% are not results.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "30b": dict(dir="model", engine="engine_v3", slots="3072",
                label="Qwen3-30B-A3B-4bit (16.0 GB)"),
    "120b": dict(dir="model-120b", engine="engine_120b", slots="600",
                 label="gpt-oss-120b-4bit (65.8 GB)"),
}


def sh(cmd, env=None, **kw):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run(cmd, cwd=HERE, env=e, text=True,
                          capture_output=True, **kw)


def environment():
    import mlx.core as mx
    info = mx.device_info()
    print("=" * 72)
    print("ENVIRONMENT")
    print(f"  mlx {mx.__version__}, python {sys.version.split()[0]}")
    print(f"  memory limit {info['max_recommended_working_set_size']/2**30:.2f} GiB"
          f", max buffer {info['max_buffer_length']/2**30:.2f} GiB")
    out = sh(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"]).stdout.split("\n")
    print(f"  {out[0]}, {int(out[1])/2**30:.0f} GB unified memory")
    load = os.getloadavg()[0]
    print(f"  load average {load:.2f}" + ("   <- machine is busy; numbers will "
                                          "be noisy" if load > 1.5 else ""))


def check_store(m):
    d = os.path.join(HERE, m["dir"])
    idx = os.path.join(d, "experts_index.json")
    if not os.path.exists(idx):
        return None, f"missing {m['dir']}/experts_index.json -- run the repack first"
    i = json.load(open(idx))
    bin_ = os.path.join(d, "experts.bin")
    want = i["n_layers"] * i["n_experts"] * i["blob_bytes"]
    got = os.path.getsize(bin_)
    if got != want:
        return None, f"{m['dir']}/experts.bin is {got} bytes, expected {want}"
    return i, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--model", choices=list(MODELS) + ["all"], default="all")
    a = ap.parse_args()
    environment()

    todo = list(MODELS) if a.model == "all" else [a.model]
    results = {}
    for key in todo:
        m = MODELS[key]
        print("=" * 72)
        print(f"{key.upper()}: {m['label']}")
        idx, err = check_store(m)
        if err:
            print(f"  SKIPPED -- {err}")
            continue
        print(f"  store {os.path.getsize(os.path.join(HERE, m['dir'], 'experts.bin'))/2**30:.1f} GiB, "
              f"{idx['n_layers']}x{idx['n_experts']} experts, "
              f"blob {idx['blob_bytes']/1e6:.2f} MB, top-{idx.get('top_k','?')}")

        print("\n  [1/2] verifying the pool byte-for-byte")
        v = sh([sys.executable, "verify_pool.py", m["dir"], m["slots"]])
        tail = [l for l in v.stdout.strip().split("\n") if l][-4:]
        for l in tail:
            print("    " + l.strip())
        if v.returncode != 0:
            print("    POOL VERIFICATION FAILED -- not timing a broken pool")
            results[key] = dict(status="pool verification failed")
            continue

        print("\n  [2/2] cold benchmark")
        t0 = time.perf_counter()
        b = sh([sys.executable, "bench.py"],
               env=dict(MOE_COLD="1", PF_SLOTS=m["slots"], PF_DEPTH="1",
                        BENCH_ENGINE=m["engine"], BENCH_LABEL=key,
                        BENCH_TRIALS="1" if a.quick else "3",
                        BENCH_TOKENS="16" if a.quick else "32"))
        if b.returncode != 0:
            print(b.stdout[-1500:] + b.stderr[-1500:])
            results[key] = dict(status="benchmark failed")
            continue
        for l in b.stdout.strip().split("\n"):
            print("    " + l)
        med = [l for l in b.stdout.split("\n") if "median" in l]
        results[key] = dict(status="ok", line=med[0].strip() if med else "",
                            seconds=time.perf_counter() - t0)

    print("=" * 72)
    print("SUMMARY")
    for k, r in results.items():
        print(f"  {k:<5} {r.get('line') or r['status']}")
    print("\n  Spread matters more than the median here: run-to-run sd is ~15% on")
    print("  this machine, so treat anything under ~12% as no difference at all.")


if __name__ == "__main__":
    main()
