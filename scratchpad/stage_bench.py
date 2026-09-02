#!/usr/bin/env python3
"""Does a contiguous read + memcpy beat a 13-segment scatter read?

Scatter costs 10-36% depending on queue depth. A contiguous read into a staging
buffer runs at full rate but adds a 14 MB memcpy. Net?
"""
import fcntl, os, random, statistics as S, threading, time

F_NOCACHE = 48
PATH = "/Users/agastya/Desktop/moe-stream/model-120b/experts.bin"
BLOB = 14_024_704
SEG = [4147200, 259200, 259200, 5760] * 3 + [10624]
NREAD = int(os.environ.get("NREAD", "256"))
NBLOB = os.path.getsize(PATH) // BLOB
_seed = [0]


def openf():
    fd = os.open(PATH, os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    return fd


def run(nthread, mode):
    _seed[0] += 1
    rng = random.Random(9000 + _seed[0])
    offs = [o * BLOB for o in rng.sample(range(NBLOB), NREAD)]
    counts = [0] * nthread
    barrier = threading.Barrier(nthread + 1)

    def worker(w):
        fd = openf()
        dest = [memoryview(bytearray(s)) for s in SEG]
        stage = memoryview(bytearray(BLOB))
        barrier.wait()
        for i in range(w, len(offs), nthread):
            if mode == "scatter":
                n = os.preadv(fd, dest, offs[i])
            else:
                n = os.preadv(fd, [stage], offs[i])
                o = 0
                for d in dest:                 # memcpy into place
                    ln = len(d)
                    d[:] = stage[o:o + ln]
                    o += ln
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


res = {m: {t: [] for t in (1, 2, 4, 8)} for m in ("scatter", "stage")}
for rnd in range(3):
    modes = ("scatter", "stage") if rnd % 2 == 0 else ("stage", "scatter")
    for m in modes:
        for nt in (1, 2, 4, 8):
            res[m][nt].append(run(nt, m))
print(f"{BLOB/1e6:.2f} MB blobs, fresh offsets, 3 rounds, order alternated\n")
print(f"  {'mode':<26} " + " ".join(f"{t:>6}thr" for t in (1, 2, 4, 8)))
for m in ("scatter", "stage"):
    lbl = "13-seg scatter (engine)" if m == "scatter" else "contiguous + memcpy"
    print(f"  {lbl:<26} " + " ".join(f"{S.median(res[m][t]):8.2f}" for t in (1,2,4,8)))
print()
for t in (1, 2, 4, 8):
    a, b = S.median(res['scatter'][t]), S.median(res['stage'][t])
    print(f"  {t} thread: stage/scatter = {b/a:.2f}x")
