#!/usr/bin/env python3
"""Is the engine's residual read-rate gap (3.05 GB/s vs 3.4-3.8 contiguous) GIL
contention with MLX dispatch?

HANDOFF already answered "no" once -- workers sustained 3.50 GB/s while the main
thread looped on mx.eval, vs 3.49 idle. But that was measured BEFORE MOE_STAGE,
when a read was a bare preadv (which releases the GIL for its whole duration).
With MOE_STAGE=1 each read now ends with a 14 MB copy out of the staging buffer
into 13 ctypes segments, written as `d[:] = buf[o:o+ln]` on memoryviews --
and CPython's memoryview slice assignment does NOT release the GIL. So every
blob now holds the GIL for a memcpy, and 4 workers plus the main thread's MLX
dispatch all contend for it. That is a different question and it is open.

Rule 1: real destinations. A bytearray destination once predicted 1.2x for a
change that was 35x slower, so this drives the REAL ExpertPool's `_read` against
the REAL experts.bin, with the real ctypes segments.

Arms:
  copy=mv      memoryview slice assignment      (what the engine ships)
  copy=memmove ctypes.memmove                   (ctypes foreign calls drop the GIL)
  copy=none    MOE_STAGE=0, preadv scatters     (no copy at all, but 13 segments)

Main thread:
  main=idle    sleeps
  main=mlx     loops on a small matmul + mx.eval, like the engine's router
"""
import ctypes
import os
import random
import sys
import threading
import time

sys.path.insert(0, "/Users/agastya/Desktop/moe-stream")

MODEL = os.environ.get("MODEL", "120b")
NSLOTS = int(os.environ.get("SLOTS", "64"))
NREAD = int(os.environ.get("NREAD", "192"))
NTHREAD = int(os.environ.get("THREADS", "4"))
COPY = os.environ.get("COPY", "mv")
MAIN = os.environ.get("MAIN", "idle")

os.environ["MOE_COLD"] = "1"
os.environ["PF_SLOTS"] = str(NSLOTS)
os.environ["MOE_STAGE"] = "0" if COPY == "none" else "1"

import mlx.core as mx
import engine_120b as E120
import engine_v3 as E30

E = E120 if MODEL == "120b" else E30
DIR = os.path.expanduser(
    f"~/Desktop/moe-stream/model-120b" if MODEL == "120b" else "~/Desktop/moe-stream/model")


def build_pool():
    return E.ExpertPool(DIR, n_slots=NSLOTS, n_workers=1)


def patch_memmove(pool):
    """Replace the memoryview slice copy with ctypes.memmove.

    ctypes releases the GIL around a foreign function call, so the 14 MB copy
    stops serialising against the other workers and against MLX dispatch.
    """
    dests = []
    for s in pool.slots:                       # ctypes segments, per slot
        addrs = []
        for seg in s:
            if isinstance(seg, memoryview):    # the padding scratch
                addrs.append((ctypes.addressof(
                    (ctypes.c_char * len(seg)).from_buffer(seg)), len(seg)))
            else:
                addrs.append((ctypes.addressof(seg), len(seg)))
        dests.append(addrs)
    pool._dests = dests
    memmove = ctypes.memmove

    def _read(slot, layer, expert, _p=pool):
        with _p._iolock:
            if _p._inflight == 0:
                _p._busy_t0 = time.perf_counter()
            _p._inflight += 1
            _p.depth_sum += _p._inflight
            _p.depth_n += 1
        try:
            off = (layer * _p.n_experts + expert) * _p.blob
            buf = getattr(_p._tls, "stage", None)
            if buf is None:
                buf = _p._tls.stage = (ctypes.c_char * _p.blob)()
                _p._tls.stage_mv = memoryview(buf).cast("B")
                _p._tls.stage_addr = ctypes.addressof(buf)
            got = os.preadv(_p.fd, [_p._tls.stage_mv], off)
            if got != _p.blob:
                raise IOError(f"short read {got}")
            src = _p._tls.stage_addr
            o = 0
            for addr, ln in _p._dests[slot]:
                memmove(addr, src + o, ln)
                o += ln
            _p.bytes_read += _p.blob
        finally:
            with _p._iolock:
                _p._inflight -= 1
                if _p._inflight == 0:
                    _p.busy_s += time.perf_counter() - _p._busy_t0
    pool._read = _read


def main_mlx(stop):
    """Approximate the engine's main thread: a router-sized matmul per sync."""
    a = mx.random.normal((1, 2048)).astype(mx.bfloat16)
    w = mx.random.normal((2048, 128)).astype(mx.bfloat16)
    mx.eval(a, w)
    n = 0
    while not stop.is_set():
        y = mx.argpartition(mx.softmax(a @ w, axis=-1), kth=-8, axis=-1)[..., -8:]
        mx.eval(y)
        n += 1
    return n


def run(pool, rng):
    """NTHREAD workers each pulling from a shared queue of fresh random blobs.

    Fresh offsets every run -- F_NOCACHE is not sufficient (HANDOFF: a fixed
    offset list once reported 5.79 GB/s, above the physical ceiling).
    """
    n_ex, n_ly = pool.n_experts, pool.n_layers
    jobs = [(rng.randrange(n_ly), rng.randrange(n_ex)) for _ in range(NREAD)]
    cursor = [0]
    clock = threading.Lock()

    def worker(wid):
        while True:
            with clock:
                i = cursor[0]
                cursor[0] += 1
            if i >= len(jobs):
                return
            layer, expert = jobs[i]
            pool._read(i % NSLOTS, layer, expert)

    b0 = pool.bytes_read
    ths = [threading.Thread(target=worker, args=(w,)) for w in range(NTHREAD)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    dt = time.perf_counter() - t0
    return (pool.bytes_read - b0) / dt / 1e9, dt


if __name__ == "__main__":
    pool = build_pool()
    if COPY == "memmove":
        patch_memmove(pool)
    rng = random.Random(int(os.environ.get("SEED", "0")))

    stop = threading.Event()
    spins = [0]
    if MAIN == "mlx":
        th = threading.Thread(target=lambda: spins.__setitem__(0, main_mlx(stop)))
        th.start()

    rate, dt = run(pool, rng)
    stop.set()
    if MAIN == "mlx":
        th.join()

    print(f"MODEL={MODEL} COPY={COPY:<8} MAIN={MAIN:<5} THREADS={NTHREAD}  "
          f"{rate:6.3f} GB/s   ({dt:.2f}s, {NREAD} blobs, "
          f"main-thread evals {spins[0]})")
