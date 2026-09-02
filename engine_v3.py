#!/usr/bin/env python3
"""
Streaming MoE engine with prefetch + LFU residency.

The synchronous engine spends 42% of wall time blocked on reads that could
have been issued layers earlier. Phase 1 measured the fix: applying layer
L+d's router to layer L's hidden state predicts L+d's experts far above
chance, with no training and no extra weights (residual stream means the
gate inputs stay similar across layers). At d=2 that was ~79% recall at zero
over-fetch, and depth 2 also puts 8 reads in flight, which Phase 0b showed is
where this SSD stops gaining from queue depth.

Two structural changes from engine.py:

  Slot pool. Rather than staging per layer, experts live in a pool of fixed
  slots. gather_qmm's rhs_indices can address any slot, so one strided view
  over the whole pool serves every layer and the views are built once instead
  of per fetch. Caching and prefetching then become the same mechanism: a
  prefetch is just an early fill of a slot the next layer will ask for.

  LFU eviction, not LRU. Phase 1 found LRU returns *zero* hits below 20%
  capacity here -- expert access is cyclic (layer 0..47, repeat), which is
  LRU's pathological case: it evicts every entry exactly before it is needed
  again. LFU is immune and won at every capacity that fits in memory.
"""

import ctypes
import fcntl
import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models import qwen3_moe
from mlx_lm.models.activations import swiglu
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tokenizer_utils import load as load_tokenizer

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "ext", "build"))
import mlx_zerocopy_ext as zc
import kernels as K

MODEL_DIR = os.path.expanduser("~/Desktop/moe-stream/model")
# 3072 slots = 8.15 GB of pool. The old default of 512 (and the 1536 the docs
# recommended) sat on the wrong side of a cliff: true cache hit rate goes
# 68.7% -> 76.4% -> 90.7% -> 97.4% across 1536/2048/2560/3072, and bytes read
# per token collapse 344 -> 32 MB, which takes the token from I/O-bound to
# compute-bound. Measured 5.63 -> 16.09 tok/s (2.86x), output bit-identical.
# Ceiling is 3236 slots: Metal's max_buffer_length is 8 GiB and MLX's shapes are
# int32, and for this blob size both land on the same number.
N_SLOTS = int(os.environ.get("PF_SLOTS", "3072"))
N_WORKERS = int(os.environ.get("PF_WORKERS", "4"))
PREFETCH_DEPTHS = [int(d) for d in os.environ.get("PF_DEPTH","1").split(",")]
PREFETCH_DEPTH = PREFETCH_DEPTHS[0]
PROF = defaultdict(float)

# Benchmark mode. experts.bin is 15 GB on a 16 GB machine, so whatever fraction
# the OS happens to be caching hands out free expert hits -- which is why
# identical code has measured anywhere from 2 to 8 tok/s depending on what else
# was resident. COLD bypasses the page cache so throughput depends only on the
# SSD and our own slot pool, both of which we control. Use COLD for every A/B;
# use warm only for the headline number a real user would see.
COLD = os.environ.get("MOE_COLD", "0") == "1"
# Fused single-dispatch expert kernel for batch-1 decode. Off by default until
# the end-to-end A/B says otherwise; in isolation it is at parity with
# gather_qmm because the block is bandwidth-bound, but in the engine the seven
# dispatches per layer interleave with the per-layer sync, and that is what
# fusion may recover.
# Speculate wider than top-k. The prefetcher predicts layer L+d's experts from
# layer L's hidden state at ~87.6% recall, and the ~12% it misses become demand
# reads that BLOCK the layer. The profile says those misses cost latency, not
# bandwidth: at 3072 slots the engine uses 0.71 GB/s of the 3.5 GB/s this SSD
# delivers, so there is ~4.9x of read headroom sitting idle. Fetching the top
# (k * PF_WIDEN) predicted experts instead of the top k spends that headroom to
# raise coverage, which matters most at SMALLER pools -- it buys back memory.
PF_WIDEN = int(os.environ.get("PF_WIDEN", "1"))
# Eviction score. THE FREQUENCY COUNTER WAS NEVER INCREMENTED: self.freq was
# declared and read but never written (engine_prefetch.py:228 still has the
# `self.freq[key] += 1` that the v3 rewrite dropped), so every candidate scored
# 0 and eviction was effectively random. "asis" reproduces that, for A/B only.
#
# Measured on the 120b, interleaved, order rotated, 3 rounds, MOE_COLD=1
# PF_SLOTS=600 PF_DEPTH=1 TOKENS=96. MB/token is near-deterministic:
#
#   asis (shipped)  1400 MB/token (1400-1400)   1.93 tok/s   cache 41% pf 46%
#   lru             1279 MB/token (1275-1280)   2.11 tok/s   cache 49% pf 39%
#   lfu             1216 MB/token (1212-1222)   2.24 tok/s   cache 52% pf 35%
#
# LFU wins: 1.15x fewer bytes, 1.16x throughput, ranges nowhere near touching.
# Why frequency and not recency, given usage is near-uniform: a prefetch hit
# STILL performs a full read, so bytes track pool ADMISSIONS, not demand hit
# rate. Retaining recurring experts means the prefetcher finds them resident
# and skips the read -- visible as prefetch hits falling 46% -> 35% while cache
# hits rise 41% -> 52%. Recency churns that set; frequency holds it.
EVICT = os.environ.get("MOE_EVICT", "lfu").lower()

# Read each blob contiguously into a staging buffer and memcpy into the pool's
# 13 component segments, instead of letting preadv scatter directly. Measured
# on this drive (14 MB blobs, fresh offsets, 3 rounds, order alternated, GB/s):
#
#   threads                  1      2      4      8
#   13-seg scatter        2.08   2.72   3.07   3.39
#   contiguous + memcpy   2.76   3.44   3.57   3.65
#   gain                  1.33x  1.27x  1.16x  1.08x
#
# The engine runs at mean queue depth ~2.8, so this is worth ~1.2x on the read
# rate. Costs one blob-sized buffer per worker thread (4 x 14 MB = 56 MB).
STAGE = os.environ.get("MOE_STAGE", "1") == "1"
USE_FUSED = os.environ.get("MOE_FUSED", "0") == "1"
FUSED_NSG = int(os.environ.get("MOE_NSG", "16"))
D_FF = 768
F_NOCACHE, F_RDAHEAD = 48, 45
_PRED = {}          # layer -> predicted expert set, for measuring real recall
_RECALL = []


class ExpertPool:
    """Fixed pool of expert slots with async fill and LFU eviction."""

    def __init__(self, model_dir, n_slots=N_SLOTS, n_workers=N_WORKERS):
        idx = json.load(open(os.path.join(model_dir, "experts_index.json")))
        self.blob = idx["blob_bytes"]
        self.n_experts = idx["n_experts"]
        self.n_layers = idx.get("n_layers", 48)
        self.top_k = idx.get("top_k", 8)
        self.group_size = idx["quantization"]["group_size"]
        self.bits = idx["quantization"]["bits"]
        self.n_slots = n_slots

        self.fd = os.open(os.path.join(model_dir, "experts.bin"), os.O_RDONLY)
        self.cold = COLD
        if self.cold:
            fcntl.fcntl(self.fd, F_NOCACHE, 1)   # no OS page cache
            fcntl.fcntl(self.fd, F_RDAHEAD, 0)   # no readahead: reads are random

        # ONE MLX ARRAY PER COMPONENT, not one buffer for the whole pool.
        #
        # The obvious layout -- a single flat uint32 buffer, one strided view per
        # component -- is silently wrong past a certain slot count, and it took a
        # corrupted model to find out. MLX shape dimensions are int32, and
        # `mx.view(uint32 -> bfloat16)` doubles the element count *without
        # checking*: at 460 slots of this model's 14.02 MB blob it returns an
        # array whose shape has wrapped to -1069285376. Every bf16 view built on
        # it (scales, biases, and gpt-oss's per-projection bias) then reads the
        # wrong memory for any slot past 2^31 / (blob/2) -- slot 307 here, slot
        # 1619 on Qwen3-30B, i.e. roughly half of each engine's default pool.
        # The uint32 weight view is unaffected, so the weights are right and only
        # the scales are wrong, which is why the output stayed superficially
        # plausible instead of turning to noise.
        #
        # Allocating each component separately removes the failure mode by
        # construction: every array is created in its final dtype and shape, no
        # bit-casting and no strides, so there is nothing left to overflow. It
        # also drops the old 613-slot ceiling, because `max_buffer_length` is a
        # per-buffer limit and the pool is no longer one buffer -- RAM becomes
        # the only cap.
        #
        # One expert is still one contiguous read: `preadv` scatters a single
        # file range across the twelve per-component destinations.
        comps = sorted(idx["components"], key=lambda c: c["offset"])
        end = 0
        for c in comps:
            if c["offset"] != end:
                raise RuntimeError(f"blob layout has a hole before {c['proj']}."
                                   f"{c['part']}: {c['offset']} != {end}")
            end += c["nbytes"]
        self.payload = end
        self.pad = self.blob - end          # page padding at the end of a blob
        if self.pad < 0:
            raise RuntimeError("components overrun the blob")

        self.parts, self.views = [], {}
        iov = [[] for _ in range(n_slots)]
        for c in comps:
            name = f"{c['proj']}.{c['part']}"
            dt = mx.uint32 if c["dtype"] == "U32" else mx.bfloat16
            arr = mx.zeros((n_slots, *c["shape"]), dtype=dt)
            mx.eval(arr)
            if arr.size >= 2 ** 31:
                raise RuntimeError(f"{name} has {arr.size} elements; MLX shapes "
                                   f"are int32. Reduce n_slots.")
            ptr = zc.data_ptr(arr)
            if ptr % zc.page_size():
                raise RuntimeError(f"{name} buffer is not page aligned")
            self.views[name] = arr
            self.parts.append((name, arr, c["nbytes"]))
            for i in range(n_slots):
                iov[i].append((ctypes.c_char * c["nbytes"])
                              .from_address(ptr + i * c["nbytes"]))
        # the tail padding still has to be read, or the read length stops being a
        # multiple of the page size, which F_NOCACHE cares about. One scratch per
        # slot rather than one shared: reads for different slots run concurrently.
        self._scratch = [bytearray(self.pad) for _ in range(n_slots)] if self.pad \
            else None
        if self.pad:
            for i in range(n_slots):
                iov[i].append(memoryview(self._scratch[i]))
        self.slots = iov
        # The segments are ctypes arrays. They satisfy the buffer protocol, so
        # preadv scatters into them fine, but `arr[:] = src` is ctypes' own
        # per-element slice assignment: 0.09 GB/s against 48.8 GB/s through a
        # memoryview. Cache byte-views once for the staging copy in _read.
        self.slots_mv = [[memoryview(seg).cast("B") for seg in s] for s in iov]
        self.pool_bytes = sum(a.nbytes for _, a, _ in self.parts)

        self.lock = threading.Lock()
        self.slot_of = {}                       # (layer, expert) -> slot
        self.key_of = [None] * n_slots
        self.freq = defaultdict(int)
        self.free = list(range(n_slots))
        self.pending = {}                       # key -> Future
        self.from_prefetch = set()              # filled early, not yet consumed
        self.pinned = set()
        self.clock = 0                          # monotonic allocation counter
        self.born = {}                          # key -> clock at insertion
        self.last_use = {}                      # key -> use_clock at last touch
        self.use_clock = 0                      # monotonic access counter
        self.pool = ThreadPoolExecutor(n_workers, thread_name_prefix="fetch")

        # per-token slot traffic: every layer acquires top_k and speculates
        # another top_k for each prefetch depth
        per_token = self.n_layers * self.top_k * (1 + len(PREFETCH_DEPTHS))
        self.needs_barrier = n_slots < per_token * 1.25
        self.n_prefetch_hit = 0
        self.n_cache_hit = 0
        self.n_miss = 0
        self.bytes_read = 0
        self.t_blocked = 0.0

        # I/O saturation telemetry. bytes_read/token_time understates the
        # achieved rate whenever the SSD is idle, and cannot tell "the device
        # is the floor" from "the device is idle half the time". These measure
        # the device only while it has work: busy_s is wall time with >=1 read
        # outstanding, and depth_sum/depth_n is the mean queue depth at issue.
        self._tls = threading.local()   # per-worker staging buffer, see _read
        self._iolock = threading.Lock()
        self._inflight = 0
        self._busy_t0 = 0.0
        self.busy_s = 0.0
        self.depth_sum = 0
        self.depth_n = 0
        self.n_prefetch_bail = 0      # _alloc said "pool exhausted"
        self.n_prefetch_issued = 0
        self.bytes_exposed = 0        # demand misses: read with nothing hiding them

    # -- internals -----------------------------------------------------------
    def _read(self, slot, layer, expert):
        with self._iolock:
            if self._inflight == 0:
                self._busy_t0 = time.perf_counter()
            self._inflight += 1
            self.depth_sum += self._inflight
            self.depth_n += 1
        try:
            off = (layer * self.n_experts + expert) * self.blob
            if STAGE:
                # One contiguous read into a per-thread staging buffer, then
                # memcpy into the 13 component segments. The scatter-gather
                # preadv the pool layout implies costs 33% at queue depth 1 and
                # 16% at 4 (measured, same offsets, same bytes); the 14 MB copy
                # costs far less than that. Buffer is per worker thread: 4
                # workers = 56 MB.
                buf = getattr(self._tls, "stage", None)
                if buf is None:
                    buf = self._tls.stage = memoryview(bytearray(self.blob))
                got = os.preadv(self.fd, [buf], off)
                if got == self.blob:
                    o = 0
                    for d in self.slots_mv[slot]:
                        ln = len(d)
                        d[:] = buf[o:o + ln]
                        o += ln
            else:
                got = os.preadv(self.fd, self.slots[slot], off)
            if got != self.blob:
                raise IOError(f"short read {got} of {self.blob}")
            self.bytes_read += self.blob
        finally:
            with self._iolock:
                self._inflight -= 1
                if self._inflight == 0:
                    self.busy_s += time.perf_counter() - self._busy_t0

    def _read_task(self, slot, layer, expert):
        """Worker body. Must drop its own pending entry, otherwise a prefetch
        whose prediction was wrong is never claimed and pins its slot for
        good -- the pool leaks itself empty within a dozen tokens."""
        try:
            self._read(slot, layer, expert)
        finally:
            with self.lock:
                self.pending.pop((layer, expert), None)
                self.from_prefetch.add((layer, expert))

    # A speculative entry arrives with freq 0, which makes it the lowest-scoring
    # candidate in the pool -- plain LFU evicts the very thing that was just
    # prefetched, before the layer that asked for it ever runs. Entries younger
    # than this many allocations are therefore protected. The window covers the
    # ~16 allocations per layer (8 acquired + 8 speculated) across the depth
    # being predicted, with slack; anything older is fair game again, so a wrong
    # guess expires instead of pinning a slot forever.
    def _protect_window(self):
        return 24 * max(1, PREFETCH_DEPTH)

    def _alloc(self, key):
        """Caller holds the lock. Returns a slot, evicting a cold one.

        Sampled eviction rather than a full scan: picking the coldest of a
        random sample of SAMPLE slots is within a hair of true LFU in hit rate
        but costs O(SAMPLE) instead of O(n_slots). The full scan was ~1.2M
        Python iterations per token at 1536 slots, all of it inside the fetch
        path. (Same trick Redis uses for its approximated LFU.)
        """
        SAMPLE = 24
        self.clock += 1
        if self.free:
            slot = self.free.pop()
        else:
            window = self._protect_window()
            victim = best = None
            n = self.n_slots
            start = self.clock % n
            seen = 0
            mode = EVICT
            # stride the pool from a rotating start so sampling stays cheap
            # and does not repeatedly examine the same neighbourhood
            for j in range(n):
                sidx = (start + j * 7919) % n
                k = self.key_of[sidx]
                if k is None or sidx in self.pinned or k in self.pending:
                    continue
                young = self.clock - self.born.get(k, 0) < window
                # .get, not [] -- self.freq is a defaultdict and indexing it
                # inside the scan would insert an entry for every slot examined
                if mode == "asis":
                    base = 0                     # reproduces the shipped code
                elif mode == "lru":
                    base = self.last_use.get(k, 0)
                else:
                    base = self.freq.get(k, 0)
                f = base + (1 << 30 if young else 0)
                if best is None or f < best:
                    best, victim = f, sidx
                seen += 1
                if seen >= SAMPLE and best is not None and best < (1 << 30):
                    break
            if victim is None:
                raise RuntimeError("pool exhausted: every slot pinned or pending")
            vk = self.key_of[victim]
            self.from_prefetch.discard(vk)
            self.born.pop(vk, None)
            self.last_use.pop(vk, None)
            self.freq.pop(vk, None)
            del self.slot_of[vk]
            slot = victim
        self.key_of[slot] = key
        self.slot_of[key] = slot
        self.born[key] = self.clock
        # Deliberately NOT a use. An entry arrives here speculatively, and ~18%
        # of prefetches are wrong; stamping last_use on insertion made every
        # wrong guess look maximally recent and pinned it until it aged out,
        # which is why the first LRU attempt read MORE bytes than LFU (1519 vs
        # 1455 MB/token). The young-entry window already protects the entry
        # long enough for the layer that asked for it to run; after that a
        # never-used entry sorts to last_use 0 and is the first to go.
        return slot

    def _touch(self, key):
        """Record a use. Caller holds the lock."""
        self.use_clock += 1
        self.last_use[key] = self.use_clock
        self.freq[key] += 1

    # -- public --------------------------------------------------------------
    def prefetch(self, layer, experts):
        """Non-blocking. Fill slots for experts a later layer will probably want."""
        for e in experts:
            key = (layer, int(e))
            with self.lock:
                if key in self.slot_of or key in self.pending:
                    continue
                try:
                    slot = self._alloc(key)
                except RuntimeError:
                    self.n_prefetch_bail += 1
                    return                      # pool busy; the miss path covers it
                self.n_prefetch_issued += 1
                self.pending[key] = self.pool.submit(self._read_task, slot, *key)

    def acquire(self, layer, experts):
        """Blocking. Returns the slot holding each expert.

        Misses are issued to the worker pool together rather than pread in a
        serial loop -- Phase 0b measured this SSD gaining ~35% from queue depth
        2-4, which a one-at-a-time loop can never reach.
        """
        n = len(experts)
        slots = [None] * n
        waits = []            # (i, key, future) prefetches still in flight
        issue = []            # (i, key, slot) misses to read now

        for i, e in enumerate(experts):
            key = (layer, int(e))
            with self.lock:
                fut = self.pending.get(key)
                slot = self.slot_of.get(key)
                if fut is None and slot is None:
                    slot = self._alloc(key)
                    self.pinned.add(slot)
                    self._touch(key)        # a demand miss IS a use
                    issue.append((i, key, slot))
                elif slot is not None:
                    self.pinned.add(slot)
                    self._touch(key)
            if fut is not None:
                waits.append((i, key, fut))
            elif slot is not None and not (issue and issue[-1][0] == i):
                early = key in self.from_prefetch
                with self.lock:
                    self.from_prefetch.discard(key)
                self.n_prefetch_hit += 1 if early else 0
                self.n_cache_hit += 0 if early else 1
                slots[i] = slot
            if issue and issue[-1][0] == i:
                slots[i] = issue[-1][2]

        t0 = time.perf_counter()
        self.bytes_exposed += len(issue) * self.blob
        futs = [self.pool.submit(self._read, slot, *key) for _, key, slot in issue]
        for i, key, fut in waits:
            fut.result()
            with self.lock:
                self.pending.pop(key, None)
                self.from_prefetch.discard(key)
                slots[i] = self.slot_of[key]
                self.pinned.add(slots[i])
                self._touch(key)
            self.n_prefetch_hit += 1
        for f in futs:
            f.result()
        self.n_miss += len(issue)
        self.t_blocked += time.perf_counter() - t0

        with self.lock:
            self.pinned = set(s for s in slots if s is not None)
        return slots

    def stats(self):
        tot = self.n_prefetch_hit + self.n_cache_hit + self.n_miss
        return dict(total=tot, prefetch=self.n_prefetch_hit,
                    cache=self.n_cache_hit, miss=self.n_miss,
                    blocked=self.t_blocked, gb=self.bytes_read / 2**30,
                    busy_s=self.busy_s,
                    depth=self.depth_sum / max(1, self.depth_n),
                    bail=self.n_prefetch_bail, issued=self.n_prefetch_issued,
                    exposed_gb=self.bytes_exposed / 2**30)


def top_k_experts(gate_module, x, k, norm):
    g = mx.softmax(gate_module(x), axis=-1, precise=True)
    inds = mx.argpartition(g, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(g, inds, axis=-1)
    if norm:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    return inds, scores


def top_k_indices(gate_module, x, k):
    """Indices only, for speculation, where the scores are thrown away.

    softmax is monotone, so the top k of the logits are the top k of the
    probabilities -- the softmax and the take_along_axis behind it exist only
    to produce scores the prefetcher never looks at.
    """
    return mx.argpartition(gate_module(x), kth=-k, axis=-1)[..., -k:]


# Kept outside the module tree on purpose. Assigning a list of Modules as an
# attribute makes MLX adopt them as submodules, so every block would become a
# child of every other block and walking model.parameters() would blow up.
_BLOCKS = []
_POOL = None


def streaming_call(self, x):
    pool, layer, blocks = _POOL, self._layer, _BLOCKS
    k = self.top_k

    # Route and speculate, then take ONE sync for both.
    #
    # Routing and speculation both read this layer's hidden state, so their
    # indices are ready at the same moment -- but the engine used to eval each
    # separately, paying two CPU-GPU round trips per layer, 96 per token. The
    # arithmetic in a router is a [1,2048]x[2048,128] matmul, 262K MACs;
    # measured in isolation it is 20.2 us with the sync amortised and 209.8 us
    # with a sync, so ~190 us of every invocation is the round trip and
    # nothing else. Issuing both and evaluating once halves the count.
    _t = time.perf_counter()
    inds, scores = top_k_experts(self.gate, x, k, self.norm_topk_prob)
    preds = []
    for d in PREFETCH_DEPTHS:
        tgt = layer + d
        if tgt >= len(blocks):
            continue
        kp = min(blocks[tgt].top_k * PF_WIDEN, pool.n_experts)
        preds.append((tgt, d, top_k_indices(blocks[tgt].gate, x, kp)))
    mx.eval(inds, *(p for _, _, p in preds))
    PROF["router"] += time.perf_counter() - _t

    # Speculate for layer L+d using this layer's hidden state. The prediction
    # is free: it reuses a router that is already resident, and a wrong guess
    # costs only a slot that the LFU pass will reclaim.
    _t = time.perf_counter()
    for tgt, d, pi in preds:
        pred = np.unique(np.array(pi, copy=False).reshape(-1))
        if d == PREFETCH_DEPTH:
            _PRED[tgt] = set(pred.tolist())      # widened set: coverage, not recall
        pool.prefetch(tgt, pred)
    PROF["predict"] += time.perf_counter() - _t

    _t = time.perf_counter()
    flat = np.array(inds, copy=False).reshape(-1)
    # At batch 1 the top-k of a single token are distinct by construction --
    # argpartition returns k different positions -- so np.unique is provably a
    # no-op here beyond sorting, and acquire does not care about order. Measured
    # at ~84 us per layer for the convert+unique pair, ~4 ms/token. Prefill
    # still needs the real thing, where one expert genuinely repeats across
    # tokens in the batch.
    if flat.size == k:
        uniq, inverse = flat, None
    else:
        uniq, inverse = np.unique(flat, return_inverse=True)
    PROF["bookkeep"] += time.perf_counter() - _t

    # how good was the guess made for this layer PREFETCH_DEPTH layers ago?
    if layer in _PRED:
        want = set(uniq.tolist())
        _RECALL.append(len(want & _PRED.pop(layer)) / len(want))
    _t = time.perf_counter()
    slots = pool.acquire(layer, uniq)
    PROF["acquire"] += time.perf_counter() - _t
    _t = time.perf_counter()
    sl = np.asarray(slots, dtype=np.uint32)
    idx = mx.array((sl if inverse is None else sl[inverse]).reshape(inds.shape))
    PROF["bookkeep"] += time.perf_counter() - _t

    _t = time.perf_counter()
    # Batch-1 decode goes through the fused kernel: one dispatch for the whole
    # expert block instead of seven (3 gather_qmm + SwiGLU + the weighted
    # reduction), reading each expert's blob straight out of the slot pool and
    # keeping the [8,768] intermediate in threadgroup memory. Prefill still uses
    # gather_qmm -- the kernel is a batch-1 kernel, and prefill is where MLX's
    # batched path is genuinely better.
    if USE_FUSED and x.size == x.shape[-1]:
        # The fused kernel indexes one flat uint32 pool by byte offsets inside a
        # blob. The pool is now one array per component (see ExpertPool.__init__
        # -- the flat layout silently corrupted every slot past 2^31/(blob/2)),
        # so the kernel's addressing no longer describes the buffer it is given.
        # It needs per-component bindings before it can be re-enabled; running it
        # against the new layout would read the wrong memory.
        raise NotImplementedError(
            "MOE_FUSED needs updating for the per-component pool layout; "
            "kernels.py still assumes one flat blob-strided buffer")
        slot_arr = np.asarray(slots, dtype=np.uint32)
        if inverse is not None:
            slot_arr = slot_arr[inverse]
        y = K.moe_expert_mlp(
            pool.staging,
            mx.array(slot_arr),
            x.reshape(-1),
            scores.reshape(-1).astype(mx.float32),
            d_model=x.shape[-1], d_ff=D_FF,
            group_size=pool.group_size, nsg=FUSED_NSG)
        y = y.astype(x.dtype).reshape(x.shape)
        if pool.needs_barrier:
            mx.eval(y)
        PROF["expert_gemm"] += time.perf_counter() - _t
        return y

    xe = mx.expand_dims(x, (-2, -3))
    v = pool.views
    common = dict(rhs_indices=idx, transpose=True,
                  group_size=pool.group_size, bits=pool.bits)
    x_up = mx.gather_qmm(xe, v["up_proj.weight"], scales=v["up_proj.scales"],
                         biases=v["up_proj.biases"], **common)
    x_gate = mx.gather_qmm(xe, v["gate_proj.weight"],
                           scales=v["gate_proj.scales"],
                           biases=v["gate_proj.biases"], **common)
    h = swiglu(x_gate, x_up)
    y = mx.gather_qmm(h, v["down_proj.weight"], scales=v["down_proj.scales"],
                      biases=v["down_proj.biases"], **common)
    y = (y.squeeze(-2) * scores[..., None]).sum(axis=-2)
    # This barrier exists only because a slot could be refilled before the GPU
    # reads it. When the pool is larger than one token's entire slot traffic
    # (experts acquired + speculated, over every layer) no slot can be recycled
    # within a token, so the barrier is pure sync overhead -- 48 GPU round
    # trips per token buying nothing.
    if pool.needs_barrier:
        mx.eval(y)
    PROF["expert_gemm"] += time.perf_counter() - _t
    return y


def load_engine(model_dir=MODEL_DIR, n_slots=N_SLOTS):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    model = qwen3_moe.Model(qwen3_moe.ModelArgs.from_dict(cfg))
    weights = mx.load(os.path.join(model_dir, "nonexpert.safetensors"))
    q = cfg["quantization"]

    def class_predicate(p, m):
        if p in q:
            return q[p]
        if not hasattr(m, "to_quantized"):
            return False
        return f"{p}.scales" in weights

    nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                mode=q.get("mode", "affine"), class_predicate=class_predicate)

    global _POOL
    _POOL = ExpertPool(model_dir, n_slots=n_slots)
    qwen3_moe.Qwen3MoeSparseMoeBlock.__call__ = streaming_call
    _BLOCKS.clear()
    _BLOCKS.extend(l.mlp for l in model.model.layers)
    for i, blk in enumerate(_BLOCKS):
        blk._layer = i
        del blk.switch_mlp
    pool = _POOL

    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    model.eval()
    return model, load_tokenizer(Path(model_dir)), pool


def main():
    prompt = "Explain why mixture-of-experts models are memory-bandwidth bound."
    max_tokens = 16

    model, tok, pool = load_engine()
    print(f"resident : {mx.get_active_memory()/2**30:.2f} GB "
          f"(pool {pool.pool_bytes/2**30:.2f} GB, {pool.n_slots} slots)")
    print(f"prefetch : depth {PREFETCH_DEPTH}, {N_WORKERS} workers")

    ids = tok.encode(prompt)
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    logits = model(mx.array(ids)[None], cache=cache)
    y = mx.argmax(logits[:, -1], axis=-1)
    mx.eval(y)
    print(f"\nprefill  : {len(ids)} tokens in {time.perf_counter()-t0:.1f}s")

    s0 = pool.stats()
    PROF.clear()          # prefill is a different workload; measure decode only
    out, times = [], []
    for i in range(max_tokens):
        t = time.perf_counter()
        logits = model(y[None], cache=cache)
        y = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(y)
        dt = time.perf_counter() - t
        times.append(dt)
        out.append(y.item())
        print(f"  tok {i+1:>2}  {dt*1e3:>7.0f} ms  {1/dt:>5.2f} tok/s")

    s = pool.stats()
    n = s["total"] - s0["total"]
    pf = s["prefetch"] - s0["prefetch"]
    ch = s["cache"] - s0["cache"]
    ms = s["miss"] - s0["miss"]
    blocked = s["blocked"] - s0["blocked"]
    steady = times[2:] or times
    tps = len(steady) / sum(steady)
    total = sum(times)

    print(f"\ntext: {tok.decode(out)!r}")
    print("\n" + "=" * 64)
    print(f"PREFETCHING ENGINE: {tps:.2f} tok/s   "
          f"(synchronous baseline was 1.54)")
    print(f"  median        {sorted(times)[len(times)//2]*1e3:,.0f} ms/token")
    print(f"  expert reqs   {n:,}")
    print(f"    prefetched  {pf/n*100:5.1f}%")
    print(f"    cache hit   {ch/n*100:5.1f}%")
    print(f"    miss        {ms/n*100:5.1f}%")
    print(f"  blocked on io {blocked/total*100:5.1f}% of wall "
          f"(was 42%)")
    print(f"  resident      {mx.get_active_memory()/2**30:.2f} GB")
    if _RECALL:
        import statistics
        print(f"  PREDICT RECALL {statistics.mean(_RECALL)*100:5.1f}% "
              f"(depth {PREFETCH_DEPTH}, n={len(_RECALL):,})")
    tot_tok = sum(times)
    acct = sum(PROF.values())
    print(f"\n  --- where a token goes (total {tot_tok/len(times)*1e3:.0f} ms) ---")
    for kname, v in sorted(PROF.items(), key=lambda kv: -kv[1]):
        print(f"    {kname:<14}{v/tot_tok*100:5.1f}%   {v/len(times)*1e3:7.1f} ms/tok")
    print(f"    {'attn+other':<14}{(tot_tok-acct)/tot_tok*100:5.1f}%   "
          f"{(tot_tok-acct)/len(times)*1e3:7.1f} ms/tok")
    print("=" * 64)


if __name__ == "__main__":
    main()
