#!/usr/bin/env python3
"""Does 13-segment scatter-gather cost throughput vs one contiguous read?

The engine preadv's each 14 MB expert blob into 13 separate destination
segments (3 projections x weight/scales/biases/bias, plus padding). fd_bench
measured the 3.48 GB/s ceiling with a SINGLE buffer. Same bytes, same offsets,
only the destination layout differs.
"""
import fcntl, os, random, threading, time

F_NOCACHE = 48
PATH = "/Users/agastya/Desktop/moe-stream/model-120b/experts.bin"
BLOB = 14_024_704
ENGINE_SEG = [4147200, 259200, 259200, 5760] * 3 + [10624]
NREAD = int(os.environ.get("NREAD", "192"))


def openf():
    fd = os.open(PATH, os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    return fd


def make_iov(segs):
    return [memoryview(bytearray(s)) for s in segs]


def run(nthread, segs, nread=NREAD):
    nblob = os.path.getsize(PATH) // BLOB
    rng = random.Random(1234)
    offs = [rng.randrange(nblob) * BLOB for _ in range(nread)]
    counts = [0] * nthread
    barrier = threading.Barrier(nthread + 1)

    def worker(w):
        fd = openf()
        iov = make_iov(segs)
        barrier.wait()
        for i in range(w, len(offs), nthread):
            counts[w] += os.preadv(fd, iov, offs[i])
        os.close(fd)

    ts = [threading.Thread(target=worker, args=(w,)) for w in range(nthread)]
    for t in ts: t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in ts: t.join()
    return sum(counts) / (time.perf_counter() - t0) / 1e9


layouts = [("1 segment (contiguous)", [BLOB]),
           ("2 segments", [BLOB // 2, BLOB - BLOB // 2]),
           ("4 segments", [BLOB // 4] * 3 + [BLOB - 3 * (BLOB // 4)]),
           ("13 segments (engine)", ENGINE_SEG)]
print(f"{BLOB/1e6:.2f} MB blobs, F_NOCACHE, random offsets, GB/s\n")
print(f"  {'layout':<24} {'1 thr':>7} {'2 thr':>7} {'4 thr':>7} {'8 thr':>7}")
for name, segs in layouts:
    assert sum(segs) == BLOB, (name, sum(segs))
    r = [run(nt, segs) for nt in (1, 2, 4, 8)]
    print(f"  {name:<24} " + " ".join(f"{v:6.2f} " for v in r))
