#!/usr/bin/env python3
"""Scatter-gather vs contiguous, with the caching confound removed.

v1 reused the same 192 offsets across all four layouts in a fixed order, so the
last layout read blobs the earlier ones had already touched -- and reported
5.79 GB/s, above the single-buffer device ceiling, which is not physical.
Here every run draws FRESH offsets, layout order is shuffled per round, and
short reads are fatal rather than silently deflating the byte count.
"""
import fcntl, os, random, threading, time

F_NOCACHE = 48
PATH = "/Users/agastya/Desktop/moe-stream/model-120b/experts.bin"
BLOB = 14_024_704
ENGINE_SEG = [4147200, 259200, 259200, 5760] * 3 + [10624]
NREAD = int(os.environ.get("NREAD", "256"))
NBLOB = os.path.getsize(PATH) // BLOB
_seed = [0]


def openf():
    fd = os.open(PATH, os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    return fd


def run(nthread, segs, nread=NREAD):
    _seed[0] += 1
    rng = random.Random(9000 + _seed[0])
    offs = rng.sample(range(NBLOB), nread)          # distinct blobs, fresh
    offs = [o * BLOB for o in offs]
    counts = [0] * nthread
    barrier = threading.Barrier(nthread + 1)

    def worker(w):
        fd = openf()
        iov = [memoryview(bytearray(s)) for s in segs]
        barrier.wait()
        for i in range(w, len(offs), nthread):
            n = os.preadv(fd, iov, offs[i])
            if n != BLOB:
                raise IOError(f"short read {n}")
            counts[w] += n
        os.close(fd)

    ts = [threading.Thread(target=worker, args=(w,)) for w in range(nthread)]
    for t in ts: t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in ts: t.join()
    return sum(counts) / (time.perf_counter() - t0) / 1e9


layouts = [("1 segment (contiguous)", [BLOB]),
           ("13 segments (engine)", ENGINE_SEG)]
threads = (1, 2, 4, 8)
res = {n: {t: [] for t in threads} for n, _ in layouts}
for rnd in range(3):
    order = layouts[:] if rnd % 2 == 0 else layouts[::-1]
    for name, segs in order:
        for nt in threads:
            res[name][nt].append(run(nt, segs))
print(f"{BLOB/1e6:.2f} MB blobs, fresh offsets per run, 3 rounds, "
      f"order alternated. median GB/s\n")
print(f"  {'layout':<24} " + " ".join(f"{t:>6}thr" for t in threads))
import statistics as S
for name, _ in layouts:
    print(f"  {name:<24} " +
          " ".join(f"{S.median(res[name][t]):8.2f}" for t in threads))
print()
for t in threads:
    a = S.median(res['1 segment (contiguous)'][t])
    b = S.median(res['13 segments (engine)'][t])
    print(f"  {t} thread: scatter/contiguous = {b/a:.2f}x")
