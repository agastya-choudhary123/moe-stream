#!/usr/bin/env python3
"""Fast O(1)-per-access cache policy simulators for the MoE expert trace.

Every policy here is amortized O(1) per access -- the earlier sweep used an
O(capacity) scan per miss, which is why it took 44 s for 442K accesses.
"""
import heapq
from collections import OrderedDict, deque

import numpy as np

EPL = 128


def load_trace(ntok, path="acts/routes.npy"):
    r = np.array(np.load(path)[:, :ntok, :])
    L, T, K = r.shape
    gid = (np.arange(L)[:, None, None] * EPL + r.astype(np.int64))
    return np.ascontiguousarray(gid.transpose(1, 0, 2).reshape(-1)), L, T, K


# ---------------------------------------------------------------- classical

def lru(seq, cap):
    d, hits = OrderedDict(), 0
    for x in seq:
        if x in d:
            d.move_to_end(x); hits += 1
        else:
            if len(d) >= cap:
                d.popitem(last=False)
            d[x] = 1
    return hits / len(seq)


def lfu(seq, cap, protect=0):
    """O(1) LFU via frequency buckets; optional young-entry protection."""
    freq, buckets, born = {}, {}, {}
    minf, hits = 1, 0
    for i, x in enumerate(seq):
        if x in freq:
            hits += 1
            f = freq[x]; buckets[f].pop(x, None)
            if not buckets[f] and minf == f:
                minf = f + 1
            freq[x] = f + 1
            buckets.setdefault(f + 1, OrderedDict())[x] = 1
        else:
            if len(freq) >= cap:
                f = minf
                while True:
                    b = buckets.get(f)
                    if b:
                        v = None
                        for k in b:
                            if i - born[k] > protect:
                                v = k; break
                        if v is not None:
                            b.pop(v); freq.pop(v); born.pop(v)
                            break
                    f += 1
                    if f > minf + 64:          # no unprotected victim; force
                        f = minf
                        b = buckets.get(f)
                        while not b:
                            f += 1; b = buckets.get(f)
                        v = next(iter(b))
                        b.pop(v); freq.pop(v); born.pop(v)
                        break
            freq[x] = 1; born[x] = i
            buckets.setdefault(1, OrderedDict())[x] = 1
            minf = 1
    return hits / len(seq)


def sieve(seq, cap):
    """SIEVE (NSDI'24): FIFO order + a visited bit and a moving hand."""
    d = OrderedDict()          # x -> visited
    hits = 0
    order = deque()            # eviction scan order (oldest first)
    for x in seq:
        if x in d:
            d[x] = 1; hits += 1
        else:
            if len(d) >= cap:
                while True:
                    v = order.popleft()
                    if v not in d:
                        continue
                    if d[v] == 0:
                        del d[v]; break
                    d[v] = 0; order.append(v)
            d[x] = 0; order.append(x)
    return hits / len(seq)


def s3fifo(seq, cap, sratio=0.10):
    """S3-FIFO: small probationary FIFO + main FIFO + ghost queue."""
    scap = max(1, int(cap * sratio)); mcap = cap - scap
    S, M = deque(), deque()
    G, Gq = set(), deque()
    freq, loc = {}, {}
    hits = 0
    for x in seq:
        if x in loc:
            freq[x] = min(3, freq.get(x, 0) + 1); hits += 1; continue
        if x in G:
            while len(M) >= mcap:
                y = M.popleft()
                if freq.get(y, 0) > 0:
                    freq[y] -= 1; M.append(y)
                else:
                    del loc[y]; freq.pop(y, None)
            M.append(x); loc[x] = "M"; freq[x] = 0
            G.discard(x)
        else:
            while len(S) >= scap:
                y = S.popleft()
                if freq.get(y, 0) > 1:
                    freq[y] = 0
                    while len(M) >= mcap:
                        z = M.popleft()
                        if freq.get(z, 0) > 0:
                            freq[z] -= 1; M.append(z)
                        else:
                            del loc[z]; freq.pop(z, None)
                    M.append(y); loc[y] = "M"
                else:
                    del loc[y]; freq.pop(y, None)
                    G.add(y); Gq.append(y)
                    while len(Gq) > mcap:
                        G.discard(Gq.popleft())
            S.append(x); loc[x] = "S"; freq[x] = 0
    return hits / len(seq)


def twoq(seq, cap, kin=0.25, kout=0.50):
    """2Q: A1in FIFO -> Am LRU, with A1out ghost FIFO."""
    a1cap = max(1, int(cap * kin)); amcap = cap - a1cap
    outcap = max(1, int(cap * kout))
    A1, Am = deque(), OrderedDict()
    A1s, Aout, Aoutq = set(), set(), deque()
    hits = 0
    for x in seq:
        if x in Am:
            Am.move_to_end(x); hits += 1; continue
        if x in A1s:
            hits += 1; continue
        if x in Aout:
            Aout.discard(x)
            if len(Am) >= amcap:
                Am.popitem(last=False)
            Am[x] = 1
        else:
            if len(A1) >= a1cap:
                y = A1.popleft(); A1s.discard(y)
                Aout.add(y); Aoutq.append(y)
                while len(Aoutq) > outcap:
                    Aout.discard(Aoutq.popleft())
            A1.append(x); A1s.add(x)
    return hits / len(seq)


def lruk(seq, cap, K=2):
    """LRU-K: evict by Kth-most-recent reference time."""
    hist, hits = {}, 0
    heap = []
    res = set()
    for i, x in enumerate(seq):
        h = hist.setdefault(x, deque(maxlen=K))
        if x in res:
            hits += 1
        else:
            if len(res) >= cap:
                while True:
                    key, v = heapq.heappop(heap)
                    if v in res:
                        hv = hist.get(v)
                        cur = hv[0] if hv and len(hv) == K else -1
                        if cur == key:
                            res.discard(v); break
                        heapq.heappush(heap, (cur, v))
            res.add(x)
        h.append(i)
        key = h[0] if len(h) == K else -1
        heapq.heappush(heap, (key, int(x)))
    return hits / len(seq)


def opt(seq, cap):
    """Belady. Offline optimum -- an upper bound no online policy can pass."""
    n = len(seq)
    nxt = np.full(n, n, dtype=np.int64)
    last = {}
    for i in range(n - 1, -1, -1):
        x = int(seq[i]); nxt[i] = last.get(x, n); last[x] = i
    res, heap, hits = set(), [], 0
    for i, x in enumerate(seq):
        x = int(x)
        if x in res:
            hits += 1
        else:
            if len(res) >= cap:
                while True:
                    negt, cand = heapq.heappop(heap)
                    if cand in res and -negt >= i:
                        res.discard(cand); break
            res.add(x)
        heapq.heappush(heap, (-int(nxt[i]), x))
    return hits / n


def opt_horizon(seq, cap, H):
    """Belady limited to a lookahead of H accesses; LRU tie-break beyond it.

    Answers: how far ahead must a predictor see to capture OPT's advantage?
    H = 144 is exactly one token (36 layers x top-4). Direct O(cap) scan per
    miss -- slow but obviously correct, and only a few H values are needed.
    """
    from collections import defaultdict
    n = len(seq)
    occ = defaultdict(list)
    for i, x in enumerate(seq):
        occ[int(x)].append(i)
    ptr = defaultdict(int)
    res, last_use, hits = set(), {}, 0
    for i in range(n):
        x = int(seq[i])
        o = occ[x]
        while ptr[x] < len(o) and o[ptr[x]] <= i:
            ptr[x] += 1
        if x in res:
            hits += 1
        else:
            if len(res) >= cap:
                best, bkey = None, None
                for v in res:
                    ov = occ[v]; p = ptr[v]
                    nu = ov[p] - i if p < len(ov) else n
                    key = (min(nu, H), -last_use[v])
                    if bkey is None or key > bkey:
                        bkey, best = key, v
                res.discard(best); del last_use[best]
            res.add(x)
        last_use[x] = i
    return hits / n


def reuse_hist(seq):
    """Distribution of reuse distance, in accesses (144 accesses = 1 token)."""
    last, out = {}, []
    for i, x in enumerate(seq):
        x = int(x)
        if x in last:
            out.append(i - last[x])
        last[x] = i
    return np.array(out)


def engine_sampled(seq, cap, mode, sample=24, window=24, stride=7919):
    """Faithful replica of engine_v3.ExpertPool._alloc.

    mode 'asis' reproduces the shipped code, where self.freq is never
    incremented so every candidate scores 0 -- i.e. random-with-protection.
    mode 'lfu'  is what the comments describe (freq actually counted).
    mode 'lru'  scores by last use instead of frequency.
    """
    from collections import defaultdict
    key_of = [None] * cap
    slot_of, freq, born, last_use = {}, defaultdict(int), {}, {}
    free = list(range(cap))
    clock = hits = 0
    for x in seq:
        x = int(x)
        if x in slot_of:
            hits += 1
            freq[x] += 1
            last_use[x] = clock
            continue
        clock += 1
        if free:
            slot = free.pop()
        else:
            start = clock % cap
            best = victim = None
            seen = 0
            for j in range(cap):
                sidx = (start + j * stride) % cap
                k = key_of[sidx]
                if k is None:
                    continue
                young = clock - born.get(k, 0) < window
                base = 0 if mode == "asis" else (
                    freq[k] if mode == "lfu" else last_use.get(k, 0))
                f = base + ((1 << 30) if young else 0)
                if best is None or f < best:
                    best, victim = f, sidx
                seen += 1
                if seen >= sample and best is not None and best < (1 << 30):
                    break
            vk = key_of[victim]
            del slot_of[vk]; born.pop(vk, None)
            slot = victim
        key_of[slot] = x; slot_of[x] = slot
        born[x] = clock; last_use[x] = clock
        freq[x] += 1
    return hits / len(seq)
