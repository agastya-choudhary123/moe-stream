#!/usr/bin/env python3
"""Learned eviction for the MoE expert pool -- imitation of Belady (LRB/Parrot).

At an eviction decision we want the resident expert whose NEXT use is furthest
away. Belady knows it; an online policy must predict it. This trains a GBDT to
regress log(time-to-next-use) from features available at runtime, then evicts
argmax over a sampled candidate set.

Protocol: trained on 5 genres, evaluated on 3 HELD-OUT genres (swebench, code,
multiling). This project has been burned three times by in-sample rank
artifacts; nothing here is scored on data the model saw.
"""
import json
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, "/private/tmp/claude-501/-Users-agastya-Desktop-moe-stream/"
                   "32819ea4-feb4-45b0-94d1-5314981e9000/scratchpad")
import cachelib as C

EPL, NHIST = 128, 4
TRAIN_G = ("wiki", "mmlu_stem", "mmlu_hum", "dolly", "qanta")
TEST_G = ("swebench", "code", "multiling")
NFEAT = 3 + NHIST + 6
NLAYER, TOPK = 36, 4
SWEEP = NLAYER * TOPK          # accesses per token: one full layer sweep


def windows_for(genres):
    m = json.load(open("acts/manifest.json"))
    return [(w["lo"], w["hi"]) for w in m["windows"] if w["genre"] in genres]


def build_trace(genres):
    r = np.load("acts/routes.npy", mmap_mode="r")
    L, _, K = r.shape
    segs = []
    for lo, hi in windows_for(genres):
        g = (np.arange(L)[:, None, None] * EPL +
             np.array(r[:, lo:hi, :]).astype(np.int64))
        segs.append(np.ascontiguousarray(g.transpose(1, 0, 2).reshape(-1)))
    return np.concatenate(segs), L, K


class Feat:
    """Incrementally maintained per-expert features, indexed by global id."""

    def __init__(self, n):
        self.last = np.full(n, -1, dtype=np.int64)
        self.hist = np.zeros((n, NHIST), dtype=np.float32)   # recent deltas
        self.nacc = np.zeros(n, dtype=np.float32)
        self.ewma = np.zeros(n, dtype=np.float32)
        self.layer = (np.arange(n) // EPL).astype(np.float32)

    def touch(self, x, i):
        if self.last[x] >= 0:
            d = i - self.last[x]
            self.hist[x, 1:] = self.hist[x, :-1]
            self.hist[x, 0] = d
            self.ewma[x] = 0.7 * self.ewma[x] + 0.3 * d if self.ewma[x] else d
        self.last[x] = i
        self.nacc[x] += 1

    def rows(self, ids, t):
        ids = np.asarray(ids)
        age = (t - self.last[ids]).astype(np.float32)
        h = self.hist[ids]
        # Sweep phase: accesses cycle layer 0..35 every token, so an expert at
        # layer l cannot be touched again for ((l - cur) mod 36) * 4 accesses.
        # That is a floor on Belady's own metric, known exactly at runtime.
        cur = (t // TOPK) % NLAYER
        phase = (self.layer[ids] - cur) % NLAYER
        out = np.empty((len(ids), NFEAT), dtype=np.float32)
        out[:, 0] = np.log1p(age)
        out[:, 1] = np.log1p(self.nacc[ids])
        out[:, 2] = self.layer[ids]
        out[:, 3:3 + NHIST] = np.log1p(h)
        out[:, 3 + NHIST] = np.log1p(self.ewma[ids])
        out[:, 4 + NHIST] = np.log1p(h.mean(1))
        out[:, 5 + NHIST] = age / (1.0 + self.ewma[ids])
        out[:, 6 + NHIST] = phase                      # deterministic floor
        out[:, 7 + NHIST] = age / SWEEP                # age measured in tokens
        out[:, 8 + NHIST] = self.nacc[ids] / (1.0 + age / SWEEP)
        return out


def make_dataset(seq, nobj, rng, probes=3):
    """Sample (features at a probe time, true time-to-next-use) pairs."""
    n = len(seq)
    nxt = np.full(n, n, dtype=np.int64)
    last = {}
    for i in range(n - 1, -1, -1):
        x = int(seq[i]); nxt[i] = last.get(x, n); last[x] = i

    f = Feat(nobj)
    X, y = [], []
    for i in range(n):
        x = int(seq[i])
        f.touch(x, i)
        nx = nxt[i]
        if nx >= n:
            continue
        gap = nx - i
        for _ in range(probes):
            t = i + rng.integers(0, gap) if gap > 1 else i
            X.append(f.rows([x], t)[0])
            y.append(np.log1p(nx - t))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def sim_learned(seq, cap, model, nobj, rng, nsample=64, warm=50_000):
    """Evict argmax predicted next-use over a random candidate sample."""
    f = Feat(nobj)
    res = []                      # list for O(1) random sampling
    pos = {}
    hits = tot = 0
    for i in range(len(seq)):
        x = int(seq[i])
        counted = i >= warm
        if counted:
            tot += 1
        if x in pos:
            if counted:
                hits += 1
        else:
            if len(res) >= cap:
                k = min(nsample, len(res))
                cand = rng.choice(len(res), k, replace=False)
                ids = np.array([res[c] for c in cand])
                score = model.predict(f.rows(ids, i))
                j = cand[int(np.argmax(score))]
                v = res[j]
                last = res.pop()
                if j < len(res):
                    res[j] = last; pos[last] = j
                del pos[v]
            pos[x] = len(res); res.append(x)
        f.touch(x, i)
    return hits / max(tot, 1)


def main():
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 600

    tr, L, K = build_trace(TRAIN_G)
    te, _, _ = build_trace(TEST_G)
    nobj = L * EPL
    print(f"train {len(tr):,} accesses ({', '.join(TRAIN_G)})")
    print(f"test  {len(te):,} accesses ({', '.join(TEST_G)})  HELD OUT\n")

    X, y = make_dataset(tr, nobj, rng)
    print(f"training set {X.shape}", flush=True)
    names = ["age", "nacc", "layer", "d1", "d2", "d3", "d4",
             "ewma", "dmean", "age_over_ewma", "phase", "age_tokens",
             "rate"]
    model = lgb.train(
        {"objective": "regression", "num_leaves": 63, "learning_rate": 0.08,
         "verbose": -1, "num_threads": 8},
        lgb.Dataset(X, label=y, feature_name=names), num_boost_round=300)

    Xt, yt = make_dataset(te[:200_000], nobj, np.random.default_rng(1))
    pred = model.predict(Xt)
    ss = 1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
    print(f"held-out R^2 on log next-use distance: {ss:.3f}", flush=True)
    imp = sorted(zip(names, model.feature_importance("gain")),
                 key=lambda z: -z[1])
    tot_g = sum(v for _, v in imp) or 1
    print("gain share:", ", ".join(f"{a}={100*b/tot_g:.0f}%" for a, b in imp),
          flush=True)

    per = L * K * 14.02
    print(f"\n{cap} slots, held-out genres only")
    print(f"  {'policy':<14} {'hit':>7} {'MB/tok':>8} {'%OPT gap closed':>16}")
    base = C.lru(te, cap)
    o = C.opt(te, cap)
    rows = [("lfu (engine)", C.lfu(te, cap, protect=144)),
            ("lru", base), ("s3fifo", C.s3fifo(te, cap)),
            ("sieve", C.sieve(te, cap)),
            ("learned", sim_learned(te, cap, model, nobj,
                                    np.random.default_rng(2))),
            ("OPT", o)]
    for n, h in rows:
        closed = (h - base) / (o - base) * 100 if o > base else 0
        print(f"  {n:<14} {100*h:6.1f}% {per*(1-h):8.0f} {closed:15.1f}%")


if __name__ == "__main__":
    main()
