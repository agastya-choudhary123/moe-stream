"""Is the 120b engine actually at the SSD's floor, or just idle a lot?

HANDOFF reads 1071 MB/token at 2.68 GB/s and concludes "pure I/O". But
bytes_read / token_time averages over stretches where the device has no work,
so it cannot distinguish "the SSD is the floor" from "the SSD is idle a third
of the time". A standalone probe measured this device at 3.46 GB/s (2 threads)
and 3.65 GB/s (MTLIO, depth 16), against 2.74 GB/s single-threaded.

This reports the rate while the device is BUSY, plus the mean queue depth at
issue and how often prefetch bailed out. If busy-rate ~= 3.4 GB/s and the busy
fraction is well under 1, the gap is exposed latency, not bandwidth.
"""
import os, sys, time

os.environ.setdefault("PF_SLOTS", "600")
import mlx.core as mx
import engine_120b as E
from mlx_lm.models.cache import make_prompt_cache

PROMPT = "Explain why mixture-of-experts models are hard to run on small machines."


def decode(model, y, cache, n):
    for _ in range(n):
        y = mx.argmax(model(y[None], cache=cache)[:, -1], axis=-1)
        mx.eval(y)
    return y


def main():
    n_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 40

    print(f"loading (PF_SLOTS={os.environ['PF_SLOTS']}, "
          f"PF_WORKERS={os.environ.get('PF_WORKERS','4')}) ...", flush=True)
    model, tok, pool = E.load_engine()

    ids = tok.encode(PROMPT)
    cache = make_prompt_cache(model)
    y = mx.argmax(model(mx.array(ids)[None], cache=cache)[:, -1], axis=-1)
    mx.eval(y)

    # warm: the pool must reach steady state or the miss rate is a cold-start
    # artifact and every number below is wrong
    y = decode(model, y, cache, 12)
    s0 = dict(pool.stats())
    t0 = time.perf_counter()
    decode(model, y, cache, n_tokens)
    wall = time.perf_counter() - t0
    s1 = pool.stats()

    d = {k: s1[k] - s0[k] for k in
         ("total", "prefetch", "cache", "miss", "gb", "busy_s",
          "bail", "issued", "exposed_gb")}
    depth = s1["depth"]

    gb, busy = d["gb"], d["busy_s"]
    print(f"\n{n_tokens} tokens in {wall:.1f}s -> {n_tokens/wall:.2f} tok/s")
    print(f"  read            {gb:.2f} GiB  ({gb*1024/n_tokens:.0f} MB/token)")
    print(f"  hit rate        {(d['prefetch']+d['cache'])/max(1,d['total'])*100:.0f}%"
          f"   (prefetch {d['prefetch']}, cache {d['cache']}, miss {d['miss']})")
    print(f"  prefetch issued {d['issued']}, bailed {d['bail']}"
          f"  ({d['bail']/max(1,d['issued']+d['bail'])*100:.0f}% of attempts)")
    print(f"  exposed (demand) {d['exposed_gb']:.2f} GiB"
          f"  = {d['exposed_gb']/max(1e-9,gb)*100:.0f}% of bytes")
    print()
    print(f"  token-averaged rate   {gb*1.0737/max(1e-9,wall):.2f} GB/s   <- what HANDOFF quotes")
    print(f"  rate while BUSY       {gb*1.0737/max(1e-9,busy):.2f} GB/s   <- the real device rate")
    print(f"  SSD busy fraction     {busy/wall*100:.0f}% of wall")
    print(f"  mean queue depth      {depth:.2f}")
    print()
    ceiling = 3.46
    achieved_busy = gb * 1.0737 / max(1e-9, busy)
    if busy / wall < 0.9:
        idle = wall - busy
        proj = gb * 1.0737 / ceiling
        print(f"  SSD idle {idle:.1f}s of {wall:.1f}s. At {ceiling} GB/s fully")
        print(f"  saturated these bytes take {proj:.1f}s -> {n_tokens/proj:.2f} tok/s "
              f"({n_tokens/proj/(n_tokens/wall):.2f}x)")
    else:
        print(f"  device is saturated at {achieved_busy:.2f} GB/s; bytes are the only lever")


if __name__ == "__main__":
    main()
