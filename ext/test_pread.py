#!/usr/bin/env python3
"""
The engine's core primitive: SSD -> GPU with no copy.

Allocate the staging buffer with MLX (a real MTLBuffer, already resident),
take its host address, and pread expert bytes directly into it. On unified
memory that address is the same memory the GPU reads, so the SSD DMAs into
GPU-visible memory and nothing is ever memcpy'd.

Verifies three things:
  1. the GPU sees bytes written by pread behind MLX's back
  2. no allocation happens per fetch (buffer is reused)
  3. throughput matches raw pread, i.e. the MLX layer costs nothing
"""

import os
import sys
import ctypes
import fcntl
import time
import random

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"))
import mlx_zerocopy_ext as zc

SCRATCH = ("/private/tmp/claude-501/-Users-agastya-Desktop/"
           "384155b9-cfbd-4fca-bcdd-3c4e637ad4e0/scratchpad")
PATH = os.path.join(SCRATCH, "ssd_testfile.bin")

F_NOCACHE, F_RDAHEAD = 48, 45
EXPERT_BYTES = 2_654_208          # Qwen3-30B-A3B expert: 2.53 MB
N_U32 = EXPERT_BYTES // 4
TRIALS = 64

mx.eval(mx.zeros(4))              # init Metal

fd = os.open(PATH, os.O_RDONLY)
fcntl.fcntl(fd, F_NOCACHE, 1)
fcntl.fcntl(fd, F_RDAHEAD, 0)
fsize = os.path.getsize(PATH)

# --- MLX-owned staging buffer, allocated once -------------------------------
staging = mx.zeros((N_U32,), dtype=mx.uint32)
mx.eval(staging)
ptr = zc.data_ptr(staging)
nb = zc.nbytes(staging)
print(f"staging buffer : {nb/2**20:.2f} MB at 0x{ptr:x}")
print(f"page aligned   : {ptr % zc.page_size() == 0}")

view = (ctypes.c_char * nb).from_address(ptr)

# --- 1. correctness ---------------------------------------------------------
off = 4096 * 16384
n = os.preadv(fd, [view], off)
print(f"\npreadv wrote   : {n:,} bytes into the MLX buffer")

truth = np.frombuffer(os.pread(fd, nb, off), dtype=np.uint32)
gpu = mx.sum(staging[:1 << 18].astype(mx.uint64))
mx.eval(gpu)
cpu = int(truth[:1 << 18].astype(np.uint64).sum())
print(f"  gpu sum {gpu.item()}")
print(f"  cpu sum {cpu}")
print(f"  GPU SEES PREAD BYTES: {gpu.item() == cpu}")

# --- 2. no allocation per fetch --------------------------------------------
mx.clear_cache()
base = mx.get_active_memory()
for _ in range(16):
    o = random.randrange(0, fsize - nb) // 16384 * 16384
    os.preadv(fd, [view], o)
    mx.eval(mx.sum(staging[:1024]))
delta = mx.get_active_memory() - base
print(f"\n16 fetches     : MLX active delta {delta/2**20:+.2f} MB "
      f"({'no allocation' if abs(delta) < 2**20 else 'ALLOCATING'})")

# --- 3. throughput vs raw pread --------------------------------------------
def timed(fn):
    offs = [random.randrange(0, fsize - nb) // 16384 * 16384
            for _ in range(TRIALS)]
    t0 = time.perf_counter()
    for o in offs:
        fn(o)
    dt = time.perf_counter() - t0
    return TRIALS * nb / dt / 2**20

bw_mlx = timed(lambda o: os.preadv(fd, [view], o))
scratch = bytearray(nb)
mv = memoryview(scratch)
bw_raw = timed(lambda o: os.preadv(fd, [mv], o))

print(f"\npread -> MLX buffer  : {bw_mlx:,.0f} MB/s")
print(f"pread -> bytearray   : {bw_raw:,.0f} MB/s")
print(f"overhead of MLX path : {(bw_raw/bw_mlx - 1)*100:+.1f}%")

# what the copying alternative would have cost
t0 = time.perf_counter()
for _ in range(8):
    a = mx.array(np.frombuffer(scratch, dtype=np.uint32))
    mx.eval(a)
t_copy = (time.perf_counter() - t0) / 8
print(f"\nmx.array() copy path : {t_copy*1e3:.1f} ms per expert "
      f"= {nb/t_copy/2**20:,.0f} MB/s")
print(f"  per token (384 experts): {t_copy*384*1e3:,.0f} ms of pure copy avoided")

os.close(fd)
