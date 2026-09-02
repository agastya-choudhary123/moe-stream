#!/usr/bin/env python3
"""Is the 3.41 GB/s ceiling the device, or a shared-fd serialization?

engine_v3 preads 14 MB expert blobs from ONE fd across 4 worker threads. If
XNU/APFS serializes anywhere per-vnode, per-thread fds would break the ceiling.
Same blob size, same offsets, same F_NOCACHE -- only the fd strategy varies.
"""
import fcntl
import os
import random
import sys
import threading
import time

F_NOCACHE = 48
PATH = "/Users/agastya/Desktop/moe-stream/model-120b/experts.bin"
BLOB = 14_024_704
NREAD = int(os.environ.get("NREAD", "192"))


def openf():
    fd = os.open(PATH, os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    return fd


def run(nthread, shared, size=BLOB, nread=NREAD):
    total = os.path.getsize(PATH)
    nblob = total // size
    rng = random.Random(1234)
    offs = [rng.randrange(nblob) * size for _ in range(nread)]
    shared_fd = openf() if shared else None
    counts = [0] * nthread
    barrier = threading.Barrier(nthread + 1)

    def worker(w):
        fd = shared_fd if shared else openf()
        buf = bytearray(size)
        mv = memoryview(buf)
        barrier.wait()
        for i in range(w, len(offs), nthread):
            n = os.preadv(fd, [mv], offs[i])
            counts[w] += n
        if not shared:
            os.close(fd)

    ts = [threading.Thread(target=worker, args=(w,)) for w in range(nthread)]
    for t in ts:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    if shared_fd is not None:
        os.close(shared_fd)
    return sum(counts) / dt / 1e9


def main():
    print(f"{PATH.split('/')[-1]}, {BLOB/1e6:.2f} MB blobs, {NREAD} reads, "
          f"F_NOCACHE, random offsets\n")
    print(f"  {'threads':>7} {'shared fd':>11} {'per-thread fd':>14} {'gain':>7}")
    for nt in (1, 2, 4, 8, 12):
        a = run(nt, True)
        b = run(nt, False)
        print(f"  {nt:7d} {a:10.2f}G {b:13.2f}G {b/a:6.2f}x")

    print(f"\n  read-size sweep, per-thread fds, 4 threads")
    print(f"  {'size MB':>8} {'GB/s':>8}")
    for mb in (2, 4, 14, 28, 56):
        sz = mb * 1024 * 1024
        print(f"  {mb:8d} {run(4, False, size=sz, nread=max(48, NREAD//mb)):7.2f}")


if __name__ == "__main__":
    main()
