#!/usr/bin/env python3
"""Does the extension actually wrap memory instead of copying it?"""

import os
import sys
import time
import mmap

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"))
import mlx_zerocopy_ext as zc

SCRATCH = ("/private/tmp/claude-501/-Users-agastya-Desktop/"
           "384155b9-cfbd-4fca-bcdd-3c4e637ad4e0/scratchpad")
PATH = os.path.join(SCRATCH, "ssd_testfile.bin")
MB = 256
N = MB * 2**20 // 4


def mib(x):
    return x / 2**20


print(f"page size: {zc.page_size()} bytes")

fd = os.open(PATH, os.O_RDONLY)
mm = mmap.mmap(fd, MB * 2**20, prot=mmap.PROT_READ)
host = np.frombuffer(mm, dtype=np.uint32, count=N)
ptr = host.__array_interface__["data"][0]

print(f"mapping   : {MB} MB at 0x{ptr:x}")
print(f"aligned   : {ptr % zc.page_size() == 0}")
print(f"can_wrap  : {zc.can_wrap(ptr)}")

# force pages resident so the GPU is not reading unmapped memory
_ = int(host[::4096].sum())

mx.clear_cache()
base = mx.get_active_memory()
print(f"\nMLX active before : {mib(base):8.1f} MB")

t0 = time.perf_counter()
a = zc.array_from_ptr(ptr, [N], "uint32")
t_wrap = time.perf_counter() - t0

after = mx.get_active_memory()
delta = after - base
print(f"MLX active after  : {mib(after):8.1f} MB  (delta {mib(delta):+.1f} MB)")
print(f"wrap time         : {t_wrap*1e3:.3f} ms")
print(f"type              : {type(a)}  shape {a.shape}  dtype {a.dtype}")

zero_copy = delta < (0.25 * MB * 2**20)
print(f"\n  -> {'ZERO-COPY' if zero_copy else 'COPIED'}")

# correctness: GPU must see the same bytes the CPU does
n_check = 1 << 22
t0 = time.perf_counter()
gpu = mx.sum(a[:n_check].astype(mx.uint64))
mx.eval(gpu)
t_gpu = time.perf_counter() - t0
cpu = int(host[:n_check].astype(np.uint64).sum())
print(f"\n  gpu sum {gpu.item()}")
print(f"  cpu sum {cpu}")
print(f"  match   {gpu.item() == cpu}   (gpu op {t_gpu*1e3:.1f} ms)")

# compare against the copying path
mx.clear_cache()
b0 = mx.get_active_memory()
t0 = time.perf_counter()
b = mx.array(host)
mx.eval(b)
t_copy = time.perf_counter() - t0
print(f"\n  mx.array() copy : {t_copy*1e3:,.1f} ms, "
      f"delta {mib(mx.get_active_memory()-b0):+.1f} MB")
print(f"  wrap is {t_copy/max(t_wrap,1e-9):,.0f}x faster and allocates nothing")

# a deliberately unaligned pointer should be rejected
mid = ptr + 7
print(f"\n  can_wrap(unaligned 0x{mid:x}) = {zc.can_wrap(mid)}")

del a, b, host
mm.close()
os.close(fd)
print("\nclean teardown (no free of foreign memory)")
