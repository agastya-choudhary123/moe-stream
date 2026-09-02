# moe-stream — handoff

## What this is

A streaming mixture-of-experts inference engine for Apple silicon. It runs models
whose weights do not fit in RAM by keeping only the non-expert weights resident and
pulling expert weights off SSD per token, with speculative prefetch.

Currently runs **Qwen3-30B-A3B-4bit (16.0 GB) on a 16 GB M4 in ~4.6 GB resident**,
producing coherent text. Stock MLX cannot load this model on this machine at all.

## Machine

Apple M4 (base, not Pro/Max), 16 GB unified memory, 10-core GPU, ~120 GB/s memory
bandwidth. macOS. Python 3.12.0, MLX 0.32.0, mlx-lm 0.31.3. Command Line Tools only
(no full Xcode, so no `xcrun metal` shader compiler — but `mx.fast.metal_kernel`
JIT-compiles at runtime and the Metal *API* headers are in the CLT SDK).

~105 GB free disk.

## Layout

```
~/Desktop/moe-stream/
  ext/                    MLX zero-copy C++ extension (nanobind + libmlx.dylib)
    src/zerocopy.cpp        data_ptr, nbytes, page_size, array_from_ptr
    build/mlx_zerocopy_ext.cpython-312-darwin.so
  repack.py               safetensors shards -> expert-contiguous store (20 s)
  verify_repack.py        byte-exact check vs source + throughput
  model/
    experts.bin           15.19 GB, blob at (layer*128+expert)*2654208
    experts_index.json    component layout inside a blob
    nonexpert.safetensors 0.81 GB, attention/norms/embeddings/48 routers
  engine_v3.py            THE engine — concurrent acquire, sampled eviction,
                          no per-layer barrier, profiling counters. (v1
                          `engine.py` and v2 `engine_prefetch.py` were deleted
                          in the 2026-08-18 cleanup; nothing imported them and
                          v3 supersedes both.)
  kernels.py              fused single-dispatch MoE expert Metal kernel
  test_kernel.py          kernel correctness (vs fp32 reference) + isolated timing
  bench.py                interleaved multi-trial benchmark harness
                          (BENCH_ENGINE=<module> to A/B two engine versions)
  verify_pool.py          byte-verifies the slot pool, including past the old
                          int32 overflow point. Run this after touching the pool
  HANDOFF.md              this file

  activation-subspace investigation (closed negative, kept for the method):
    corpus.py             diverse local corpus, packed token windows, no download
    capture_acts.py       captures MLP-input activations for all 36 layers
    fit_basis.py          per-layer basis fitted on the training windows
    subspace_q1.py        held-out retention vs k, convergence, leave-genre-out
    subspace_q1b.py       output error vs k on real experts (+ 4-bit factors)
    subspace_floor.py     best possible error for ANY rank-k shared basis
    subspace_conditional.py  the two schemes the floor does not cover:
                          per-expert (routing-conditioned) and output-side bases
    jspace_fisher.py      Fisher at the expert-path input, by backprop through
                          the streaming engine with routing replayed
    jspace_floor.py       the floor re-priced in KL instead of activation energy
    expert_neurons.py     static neuron pruning in the intermediate dimension
    weight_probes.py      gate/up pairing, nibble entropy, permutation alignment
    sparsity_predict.py   can the contextual-sparsity row set be known a layer early
    capacity_sweep.py     interleaved slot-count sweep
    verify_quality.py     re-verification of the claims the pool bug invalidated
    subspace_q2.py        end-to-end NLL / KL / top-1 through the real engine
                          (`make_preds.py` and `pool_sim.py` deleted 2026-08-18
                          — HANDOFF already recorded them as unused)

  These scripts are KEPT deliberately even though nothing imports them: they are
  the provenance of roughly half this document's findings, and several are
  cross-imported (`subspace_q1b.py` alone is imported by nine others). Their raw
  outputs were not kept — see the cleanup note at the end.
```

  spec.py                 speculative decoding: multi-token verification with a
                          pluggable drafter (n-gram / oracle). See "Multi-token
                          verification" -- mechanism proven, drafter is the blocker
  scratchpad/             measurement instruments, kept IN THE REPO on purpose.
                          The previous session left these in a session-temp dir
                          and they nearly did not survive.
    profile_io.py           decode-only I/O profile (duty cycle, rate while busy)
    gil_io.py               is the read-rate gap GIL contention? (no)
    burst_io.py             is it burst structure? (yes)
    batch_cost.py           T(t): cost of a t-token forward, replayed sequence
    union_trace.py          union of routed experts vs tokens verified together
    oracle_spec.py          the ceiling: speculative decoding with a perfect drafter
    ab_spec.py, sweep_spec.py   interleaved A/B and drafter sweeps
    fd_bench.py, iov_bench*.py, stage_bench.py   raw drive characterisation
    cachelib.py, learned_evict.py, prefetch_sim.py, policy_sweep.py  eviction study

Phase scripts and their JSON results are in the session scratchpad:
`phase0_ssd.py`, `phase0b_chunksize.py`, `phase1_routing.py`, `phase1b_depth.py`,
`design_point.py`.

## How it works

1. **Repack.** MLX stores MoE experts stacked per layer (`switch_mlp.gate_proj.weight`
   is `[128, 768, 256]`). One expert's nine components (gate/up/down × weight/scales/
   biases) are spread across nine tensors — fetching it natively means nine reads, six
   of them 48 KB. `repack.py` rewrites so one expert is one contiguous 2.53 MB blob,
   which is exactly 162 pages, so every blob is page-aligned with zero padding.

2. **Zero-copy staging.** The staging pool is allocated by MLX, so it is a real
   `MTLBuffer` the GPU reads in place. `zc.data_ptr()` returns its host address and
   `os.preadv` writes SSD bytes directly into it. Nothing is memcpy'd; no allocation
   happens per token.

3. **Slot pool.** Experts live in N fixed slots. `gather_qmm`'s `rhs_indices` can
   address any slot, so one strided view over the whole pool serves every layer and
   views are built once. Caching and prefetching become the same mechanism.

4. **Free prefetch.** Applying layer L+d's router to layer L's hidden state predicts
   L+d's experts at ~87.6% recall (d=1) with **no training and no extra weights** —
   the residual stream keeps gate inputs similar across layers.

## THE POOL WAS SILENTLY CORRUPTING HALF OF ITSELF — found, fixed, verified

**Both engines were running a partly corrupted model at their default settings.**
Throughput numbers are unaffected. Every quality claim made before this fix is not.

`ExpertPool` kept one flat uint32 buffer and built one strided view per
component. MLX shape dimensions are int32, and `mx.view(uint32 -> bfloat16)`
doubles the element count **without checking**:

```
mx.view(staging, mx.bfloat16).shape   ->  (-1069285376,)     # 460 slots, 120b
```

Every bf16 view built on that base — `scales`, `biases`, and gpt-oss's
per-projection `bias` — reads the wrong memory for any slot past
`2^31 / (blob/2)`:

| model | blob | last good slot | default pool | share of pool corrupt |
|---|---|---|---|---|
| gpt-oss-120b | 14.02 MB | 306 | 600 | **49%** |
| Qwen3-30B | 2.53 MB | 1618 | 3072 | **47%** |

The uint32 `weight` view is *not* affected, so the 4-bit weights were always
right and only their scales were wrong. That is why nothing crashed and the
output stayed plausible instead of turning to noise — the failure mode was
designed to be invisible.

**How it surfaced.** Teacher-forced NLL on plain English prose was 6.93, which
is not a number a 120B model produces. Then the same window measured 6.93 and
6.01 on two identical runs — same input, same code. Chain that closed it:

- a single layer, run three times, is bit-identical
- the same layer after 35 other layers have run is **not** (max abs diff 39.6)
- but the pool's *bytes* are perfect: 13,824 acquisitions, 0 wrong slots
- so the bytes were right and the views were reading elsewhere → slot 306 fine,
  slot 307 wrong, exactly `2^31/(blob/2)`

**The fix: one MLX array per component instead of one buffer for the pool.**
Each array is created in its final dtype and shape, so there is no bit-cast and
no stride left to overflow. One expert is still one contiguous read — `preadv`
scatters a single file range across the twelve destinations. `verify_pool.py`
checks all of it (byte-exactness at slots either side of the old limit, view
liveness, and that no array is near 2^31); both stores PASS.

**Two consequences beyond correctness:**

- **The 613-slot ceiling is gone.** `max_buffer_length` is a *per-buffer* limit
  and the pool is no longer one buffer. 700 slots allocate and verify clean
  (9.14 GiB); RAM is now the only cap, so ~720 on this machine.
- **`MOE_FUSED` is disabled and raises.** `kernels.py` addresses one flat
  blob-strided buffer, which no longer describes the pool. It needs
  per-component bindings before it can come back. It was at parity anyway.

**What this invalidates.** Anything about *values*: the "output is bit-identical
between 1536 and 3072 slots" claim (1536 was under the limit, 3072 was not, so
they cannot have matched), the reported coherence of 120b output at 600 slots,
and any generation quality judgement at a pool above the limit. Anything about
*bytes or time* stands: hit rates, MB/token, tok/s and the capacity curve never
depended on the scales being right. Re-verify quality claims on the fixed pool.

After the fix, at 600 slots: NLL is bit-identical run to run, layer 0's output
is bit-identical run to run, it agrees with weights read straight from
`experts.bin` at rel 1.5e-2 (bf16 arithmetic plus ~0.5% routing disagreement
from a dequantized-router reference), and wiki NLL drops **6.93 -> 2.42**.

## Hard-won findings (do not re-derive these)

**Storage / hardware**
- Random reads cost nothing vs sequential at expert-tile sizes (2,215 vs 2,316 MB/s at
  13 MB, qd1). No seek penalty at multi-MB granularity.
- Queue depth roughly doubles throughput: ~2.2 GB/s at qd1, ~4 GB/s at qd4+.
- 2.53 MB reads saturate at **queue depth 2** and gain nothing beyond. Larger tiles are
  faster per byte (26 MB → 4,929 MB/s), but packing multiple experts per tile only pays
  if co-activation clusters — never measured, and marginal usage is near-uniform so it
  probably doesn't.

**Routing / caching**
- Expert usage is **nearly uniform** (Gini 0.262, zero unused experts, hottest 10% of
  experts serve 12% of accesses). There is no hot set. Load-balancing loss did its job.
- **LRU is pathological here** — literally 0% hit rate below 20% capacity. Access is
  cyclic (layer 0..47, repeat); when cycle length exceeds capacity LRU evicts every
  entry exactly before it's needed. Use LFU.
- **LFU has a cold-start trap for prefetching.** A speculative entry arrives with
  freq 0, making it the next eviction victim — you evict precisely what you just
  prefetched. Fix: protect entries younger than a bounded window (implemented via
  `born`/`clock`/`_protect_window`). This took miss rate 51% → 9.5%.
- "Reuse the previous layer's experts" scores 12.3% vs 12.5% chance — **literally
  useless**. But the hidden state that produced them predicts the next set fine.
- A trained linear probe gets 50.0% vs the free projection's 48.7% at the same budget.
  **Training buys nothing.** Don't build a learned predictor.

**MLX internals (reverse-engineered)**
- `mlx_lm.load(lazy=False)` calls `mx.eval(model.parameters())` → materializes all
  16 GB. `mlx_lm.generate` wraps decode in `wired_limit(model)` → tries to wire 16 GB.
  Both must be bypassed; neither is an MLX limitation, both are mlx_lm Python.
- `model.sanitize()` does `mx.stack()` over all 128 experts per layer per matrix. Never
  call it. Delete `switch_mlp` while it is still a lazy graph node, *before* any eval.
- Assigning a **list of Modules** as an attribute makes MLX adopt them as submodules —
  `blk._blocks = blocks` made every block a child of every other block and
  `model.parameters()` blew up until the kernel SIGKILLed it. Keep such references
  outside the module tree (module-level global).
- `mx.from_dlpack` **copies**. There is no zero-copy path in the Python API.
- `allocator::can_reuse_alien_buffer()` **segfaults** on ordinary pointers.
- `Buffer::ptr()` is the `MTL::Buffer` **object** address; `Buffer::raw_ptr()` is its
  contents. Writing to the former corrupts Metal (silent SIGSEGV).
- Wrapping foreign memory is gated by the Metal allocator — the `array(void*, ...)`
  constructor falls back to allocate+copy. **Invert it: let MLX allocate, then pread
  into its buffer.**
- `gather_qmm` accepts **strided views** (`mx.as_strided` + `mx.view` for dtype
  reinterpretation). This is what makes the expert-major blob layout viable.
- nanobind cross-module casting to `mlx.core.array` works with `NB_DOMAIN mlx`.
- CMake finds Python 3.13 by default; pass `-DPython_EXECUTABLE` explicitly.

## Current numbers

Set `MOE_COLD=1` for any comparison. See "Benchmarking" below.

**Cold baseline (reproducible):** `MOE_COLD=1 PF_SLOTS=3072 PF_DEPTH=1`
- **21.8 tok/s** in 8.40 GB resident, for a 16.0 GB model, 40 MB/token off SSD.
  Final interleaved check, 3 runs each, 32 tokens: 3072 slots median 21.80
  (21.73-21.98), 1536 slots median 6.21 (6.08-6.44) at 339 MB/token. **3.5x**,
  ranges nowhere near touching.
- Two changes got there: one sync per layer instead of two (+11%), and the pool
  size fix (2.86x). Output is bit-identical to the 1536-slot engine.
- At 1536 slots (4.60 GB resident) it is ~5.6 tok/s, if memory matters more.
- Hit rate 97.4% **true** (no disk read) at 3072 slots; 32 MB/token off the SSD.
  Beware the old "93%" framing: it counted prefetch hits, which still read a full
  2.53 MB blob. Only the *cache* share avoids the disk.
- Prediction recall **87.6%**
- Resident **4.60 GB** for a 16.0 GB model
- Repack verified byte-exact; all blob offsets page-aligned

**Pool size sweep — the earlier version of this table stopped one row short of a
cliff and drew the wrong conclusion. Corrected:**

| slots | tok/s | MB/token | true hit | resident |
|---|---|---|---|---|
| 1536 | 5.63 | 344 | 68.7% | 4.60 GB |
| 2048 | 5.04 | 259 | 76.4% | 5.87 GB |
| 2560 | 11.96 | 100 | 90.7% | 7.14 GB |
| **3072** | **16.09** | **32** | **97.4%** | **8.40 GB** |

The old sweep ran 512/1024/1536/2048, saw 2048 fail to beat 1536, and concluded
"plateau, use 1536". 2048 really is a plateau — and 2560 is 2.1x and 3072 is 2.9x.
Interleaved A/B with order alternated, 3 rounds: 1536 median 5.63 (5.37-5.98),
3072 median 16.09 (15.47-16.25). Non-overlapping, ~24x the resolution limit, and
decoded token ids are **bit-identical** between the two.

Why the cliff: what matters is the **true** hit rate (accesses served with no disk
read), not the 93% headline that counts prefetch hits — those still did a full
read. True hits go 68.7 -> 76.4 -> 90.7 -> 97.4%, so bytes/token collapse
344 -> 32 MB. At 32 MB/token the SSD needs 9 ms against a ~38 ms compute floor:
**the engine stops being I/O-bound.** That is the whole 2.9x.

~~**Ceiling is 3236 slots** (8.58 GB), from two independent limits that coincide:
`max_buffer_length` is 8 GiB, and MLX shapes are int32 so a 1D uint32 pool caps at
2^31-1 elements.~~ **Both limits are gone** — the pool is now one array per
component, not one buffer, and `max_buffer_length` is per buffer. RAM is the only
cap. Note also that the int32 limit never announced itself: past 1618 slots the
bf16 views silently read the wrong memory rather than failing. See "THE POOL WAS
SILENTLY CORRUPTING HALF OF ITSELF".

Trade-off is memory, not accuracy: 1536 slots = 4.60 GB at 5.6 tok/s, 3072 =
8.40 GB at 16.1 tok/s. `max_recommended_working_set_size` on this machine is
11.45 GB, so 3072 + weights (~9 GB) is inside the safe envelope. **Default is now
3072** in `engine_v3.py`.

## Benchmarking — read before measuring anything

`experts.bin` is 15 GB on a 16 GB machine, so whatever fraction the OS page cache
happens to hold hands out free expert hits. That made identical code measure anywhere
from 2 to 8 tok/s. An earlier 7.65 tok/s claim was retracted for this reason.

`MOE_COLD=1` opens the store with `F_NOCACHE` + no readahead, so throughput depends
only on the SSD and the slot pool. Reproducibility goes from 4x drift to ~1%.

Counterintuitively **cold is also faster than most warm runs**: on a machine this
tight, the page cache for a 15 GB file can never hold enough to help much, but it does
create eviction pressure against the 4.6 GB pool.

- **cold** — every A/B comparison, always.
- **warm** — headline number only, never for tuning.
- Use `bench.py` (warmup + median/spread). Treat overlapping spreads as no result.

## Profile (share of accounted time)

### Current profile (3072 slots, trace-replayed, ~46 ms/token)

Measured by an ADDITIVE ladder with the routing confound removed. Two earlier
attempts at this were invalid: any ablation of the expert block corrupts the
hidden state, which changes routing, which changes I/O volume and pool state
(bytes/token came out 585/120/236/8, then 0/0/65/105 across rungs). Fix: record
the expert sequence from one real run and REPLAY it, so every rung acquires the
same experts in the same order and only the compute differs.

| rung | ms/token | share |
|---|---|---|
| attention + norms + embeddings + lm_head | 9.3 | 20% |
| **+ router, top-k, and 48 syncs** | **15.0** | **33%** |
| + pool acquire and SSD reads | 0.9 | 2% |
| **+ expert GEMM** | **20.9** | **45%** |
| total | 46.1 | (21.7 tok/s) |

**I/O is finished as a problem — 2%.** The engine is compute-bound now. The two
targets are the expert GEMM and the sync path, in that order.

The expert GEMM's 20.9 ms is *GPU* bandwidth, not SSD: 8 experts x 2.65 MB per
layer, 48 layers, at the 86 GB/s ceiling = 11.8 ms of pure weight reading, and
you cannot multiply by weights you have not read. This is why **fewer bits now
buys compute** even though it no longer buys I/O — at 2-bit the same block reads
11.8 MB/layer instead of 21.2 and the floor drops with it. That is the new case
for `kernels.py`, and it is a better one than the I/O case ever was.

---

Historical, from when the engine was I/O-bound at 1536 slots (150 ms/token):

Share of **wall**, 24 tokens, cold, after the one-sync fix (150 ms/token):

```
acquire                 46.0%   69.1 ms/tok
  blocked on SSD        45.6%   68.5 ms/tok
  Python bookkeeping     0.4%    0.6 ms/tok
router (matmul + sync)  47.9%   71.9 ms/tok
everything else          3.7%
expert bytes off SSD           309 MB/token
```

Two things to read off this, both of which overturn earlier plans:

- **The Python pool hot path costs 0.4% of a token, not 50%.** Moving `acquire`
  into the C++ extension was listed here as the biggest remaining item. It is worth
  ~0.6 ms/token. Dead. What `acquire` actually does is block on the SSD.
- **The `router` timer is not measuring the router.** Its `mx.eval` waits for *all*
  queued GPU work, so it absorbs the previous layer's expert GEMM and attention. The
  router's own arithmetic is 20.2 us. Read that 47.9% as "GPU drain", not "routing".

The split between these two moves a lot run to run (a slower-I/O run measured
acquire at 69.6% of accounted time). The *ratio inside* acquire — SSD vs Python —
is a within-run measurement and is stable; trust that, not the outer split.

**I/O is the floor.** 309 MB/token at the ~3.5 GB/s this SSD delivers at queue
depth 4 is ~88 ms/token of unavoidable read time, against a 150 ms token with
68.5 ms of it exposed as blocking. Perfect overlap of I/O with GPU work would put
the ceiling near max(88, 72) ms. Fewer bytes is the only way past that.

**It is not a GIL problem.** Worker threads sustained 3.50 GB/s of `preadv` while
the main thread looped on `mx.eval`, vs 3.49 GB/s with the main thread idle — 100%.
MLX releases the GIL during eval, so prefetch genuinely overlaps compute; the
exposed blocking is demand exceeding what one layer of latency can hide, not
starved reader threads.

WARNING: the per-region timers in `engine_v3.py` are only meaningful with the
per-layer `mx.eval` barrier ON. With it off (the fast path), MLX defers GPU work and
the timers misattribute it to whichever call forces the next sync — which is how an
earlier version of this doc concluded expert GEMM was 0.4%. Either re-enable the
barrier when profiling, or microbenchmark ops in isolation — and prefer the latter,
because the barrier itself changes what you are measuring. The isolated numbers under
"On Metal kernels" were taken with R dispatches per `mx.eval`, which is the only way
to see a kernel's cost rather than the eval round trip's.

## Prefetch depth — swept, no win, keep d=1

24 runs, separate process each, configs cycled round-robin with the order flipped
every other round, `MOE_COLD=1 PF_SLOTS=1536`:

| depth | tok/s (median of 12) | MB/token | miss | recall |
|---|---|---|---|---|
| 1     | **5.22** | **357** | 6.0% | 87.7% |
| 1,2   | 5.21 | 428 (+20%) | 5.3% | 87.7% |
| 2     | 5.09 | 616* | 9.7% | 83.1% |
| 3     | 4.95 | 658* | 12.3% | 79.4% |

\* d=2 and d=3 were measured before a bookkeeping fix and their MB/token includes
prefill reads; their throughput is unaffected. d=1 and d=1,2 are post-fix.

Speculating for L+1 **and** L+2 lowers the miss rate (6.0% -> 5.3%) and lands exactly
zero throughput (ratio of medians 0.998) while reading 20% more bytes. Speculating for
L+2 *instead of* L+1 is worse on every axis, as Phase 1b's recall numbers predict.
Depth 1 is already the right answer. Extra reads do not convert into speed here
because the SSD is the constraint they are competing for.

**The 20% more bytes is the finding that matters.** Fetch depth trades bytes for
overlap, and on this workload bytes are what you are short of. Anything that raises
MB/token is moving the wrong way, and will get worse on gpt-oss-120b (1.83 GB/token
vs Qwen3's 971 MB).

## Measurement resolution on this machine — read this before any A/B

Pooled over those 24 runs: **run-to-run sd is 0.79 tok/s on a ~5.2 mean, i.e. 15%.**
With 12 runs per arm the smallest difference this rig can resolve is **~12%**.

Concretely, from the same two configs:
- rounds 1-3 (fixed order) said d=1,2 was **1.48x faster**
- rounds 4-9 (order flipped) said it was **0.86x**
- all 12 rounds pooled: **1.00x**

Both of the first two were noise, and either one written down alone would have been a
false finding. Distribution is bimodal (d=1 produced both 3.74 and 7.05 tok/s), so the
median of 3 trials inside one process is not enough; you need many *separate processes*
interleaved. Rules that follow:

- Never A/B on fewer than ~8 interleaved rounds per arm, alternating which runs first.
- A result under ~12% is not a result on this machine unless the mechanism was also
  measured in isolation and predicts it.
- `bench.py`'s 3 trials in one process is a smoke test, not an experiment.

This applies retroactively to the one-sync routing fix (+11%, 3 pairs). That number is
at the edge of resolution and the end-to-end measurement alone does not establish it.
It is kept because the *mechanism* was measured independently -- 209.8 us with a sync
vs 20.2 us amortised, one sync per layer removed, ~9 ms off a 150 ms token, ~6% -- and
mechanism and end-to-end agree. Re-verify on a quiet machine before quoting it.

## Expert traffic distribution — measured, and it constrains the design

240 decode tokens over 6 varied prompts (English prose, Python, Spanish, math,
code completion), instrumented through the real pool with the real LFU policy,
because "which experts miss" is a property of the cache, not of the model.
Counts are thin (17 accesses per (layer,expert)), so **every number below is
reported against a simulated control** — Poisson noise inflates Gini and
attenuates correlation, and without controls the raw values cannot tell "no
structure" from "not enough tokens to see it".

**1. The skew is real.**

| | observed | Poisson noise floor | excess |
|---|---|---|---|
| usage Gini | 0.492 | 0.136 | **+0.355** |
| SSD-read Gini | 0.377 | 0.206 | +0.171 |

Top 10% of experts take 29.7% of accesses, 22.2% of SSD reads. All of the
structure is *within* layers: per-layer access totals are identical (cv 0.012),
mean within-layer Gini is 0.488 against a 0.136 floor.

NOTE a discrepancy with the Phase 1 figure recorded above (Gini 0.262,
"near-uniform, there is no hot set"). This run measures higher. The two differ in
prompts, token count, and in that this one includes prefill accesses. Not
resolved — if bit allocation is going to be built on this, re-measure cleanly
first (warmup, decode only, more tokens).

**2. The skew is mostly NOT stable across prompts — this is the binding result.**

Split-half over the 6 prompts, against both controls at the same exposure:

```
observed split-half Spearman            +0.249
ceiling (perfectly stable process)      +0.885
floor   (no structure at all)           +0.000
-> the real pattern sits 28% of the way from "no structure" to "perfectly stable"
```

Bit allocation is **offline and static**, so it can only exploit structure that
repeats on every prompt. Roughly 28% of the observed skew does. The coldest-25%
set overlaps only 41% between prompt halves (25% would be chance). **A static
allocation keyed on measured traffic is therefore weakly supported** — most of
what looks like a hot/cold split is that prompt's hot/cold split, not the
model's.

**3. What a concrete allocation would buy**, using the observed reads:

| coldest X% downgraded | to | MB/token | saving | accesses hit | slots in 4.6 GB |
|---|---|---|---|---|---|
| none | 4-bit | 500 | — | 0% | 1733 |
| 50% | 2-bit | 441 | 12% | 14.7% | 2228 |
| 75% | 3-bit | 435 | 13% | 42.8% | 2079 |
| 75% | 2-bit | 370 | **26%** | **42.8%** | 2599 |

(MB/token here is inflated — no warmup, and each prompt's prefill is charged to
its 40 decode tokens. Read the *ratios*, not the absolutes.)

26% of bytes costs downgrading 75% of experts and touching 43% of accesses, and
the set chosen offline is only ~28% the right set. That is a poor trade.

**What this redirects the work toward**

- **Keying bits on *sensitivity* rather than traffic.** How much a given expert's
  output degrades when quantized is a property of its weights, so it is stable
  across prompts *by construction* — it sidesteps the entire stability problem
  that sinks the traffic-keyed version. Unmeasured. This is the version worth
  building, and it is still the novel axis.
- **The slot-count column is a second, compounding win that the byte model above
  ignores.** Smaller blobs mean more experts resident in the same 4.6 GB — 1733
  to 2599 slots at 75%/2-bit — and the pool sweep already shows more slots
  cutting the miss rate (7% at 1536, 5% at 2048). Fewer bytes per expert buys
  both less traffic per read *and* fewer reads. Worth modelling before building.
- **Uniform lower precision needs no stability assumption at all** and gets 22%
  of bytes at 3-bit. It is the honest baseline any mixed-precision scheme has to
  beat, and this project does not have that baseline yet. Get it first.

## THE BOTTLENECK, resolved — capacity, not kernels

Chain of measurements, each one killing the next hypothesis:

**1. The expert GEMM has no headroom. Confirmed twice, now under real conditions.**
Earlier benchmarks used 8 *adjacent* slots in a small pool. The engine scatters 8
slots randomly across a 1536-slot / 3.9 GB pool, and `gather_qmm` makes three
strided passes over it, so scatter was a live suspect. It is not:

```
gather_qmm  8 random slots in a 1536-slot pool   246.7 us   86.1 GB/s   11.8 ms/token
gather_qmm  8 adjacent slots, same pool          249.3 us   85.2 GB/s
fused kernel, scattered (nsg16)                  258.1 us   82.3 GB/s
   scatter penalty: 0.99x   fused vs gather: 0.96x
```

86 GB/s is the hardware ceiling for a 21 MB read (measured separately: 78-84
GB/s). The expert block is 11.8 ms of a ~160 ms token and runs at the speed of
memory. **No kernel can improve it.** Stop looking here.

**2. The compute floor is ~38 ms/token (26 tok/s).**
Ablating I/O (slots returned without reading; output invalid, stopwatch only)
gives 37.8 ms/token, and that agrees with the independent sum: 28 ms of
attention + router + Python, plus 11.8 ms of expert GEMM. So ~120 ms of every
token is storage that is not being hidden.

CAUTION: the other ablations in that experiment are confounded and were
discarded. Zeroing the expert output corrupts the hidden state, which changes
routing, which changes how many bytes get read -- MB/token came out 585 / 120 /
236 / 8 across the four configs. Any ablation in this engine must hold the
expert access sequence fixed or it measures a different workload.

**3. The SSD is already saturated at 4 threads. `PF_WORKERS` is not a lever.**
Real experts.bin, 2.53 MB blobs, random offsets, F_NOCACHE:

| threads | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| GB/s | 2.69 | 3.48 | **3.50** | 3.43 | 3.39 | 3.06 |

Peak 3.50 GB/s at 2-4 threads, and it *degrades* past 8. Phase 0b was right. The
engine moves ~344 MB/token in ~160 ms = 2.16 GB/s, i.e. 62% of peak.

**4. The 93% hit rate is misleading. The number that matters is 68.7%.**
A "prefetch hit" still performed a full SSD read -- it was just issued early. Only
*cache* hits avoid the disk. So ~31% of accesses hit storage, not 7%.

**5. Therefore: shrink the expert, and it pays twice.**
Fewer bits means fewer bytes per read AND more experts resident in the same RAM,
and the second effect is the larger one. Measured by running the real engine at
the slot count each bit width would buy inside the same fixed 3.9 GB pool
(bytes/token measured with real 4-bit blobs, then scaled by blob size; bytes are
near-deterministic where tok/s has 15% noise):

| bits | blob | slots in 3.9 GB | true hit | MB/token | read ms | floor = max(read, 38) |
|---|---|---|---|---|---|---|
| 4 | 2.65 MB | 1536 | 68.7% | 344 | 98 | **98 ms** (10 tok/s) |
| 3 | 2.06 MB | 1974 | 75.8% | 206 | 59 | **59 ms** (17 tok/s) |
| 2 | 1.47 MB | 2764 | 91.4% | 52 | 15 | **38 ms** (26 tok/s) |

Hit rate climbs steeply in this range, so the win is superlinear: 2-bit cuts
bytes/token by **6.6x**, not the 1.8x the blob size alone would suggest. At 2 bits
the engine stops being I/O-bound entirely and hits the 38 ms compute floor --
**26 tok/s, a 4x**, and at that point the expert kernel starts to matter again
because compute is finally the constraint.

`mx.gather_qmm` accepts bits 2,3,4,5,6,8, so uniform low precision needs no
kernel. Per-matrix relative error on random weights: 2-bit 0.42, 3-bit 0.20,
4-bit 0.098. 2-bit uniform is almost certainly too lossy on its own.

**Which is exactly where per-expert mixed precision earns its place** -- and on a
much stronger footing than the traffic-skew argument that the stability data
sank. The goal is not "hot experts get more bits". It is **maximum quality at a
fixed capacity target**: hold the pool at ~2764 slots, spend the byte budget
where quantization actually hurts, and key the allocation on per-expert
*sensitivity*, which is a property of the weights and therefore stable across
prompts by construction. `gather_qmm` takes one scalar `bits` per call and
cannot express it. `kernels.py` can. That is what the kernel is for.

**Practical blocker to resolve first:** the only weights on disk are already
4-bit (`mlx-community/Qwen3-30B-A3B-4bit`). Going to 3 or 2 bits from there is
double quantization and throws away quality for nothing. Doing this properly
wants the bf16 original (~60 GB, 105 GB free) quantized once, directly to the
target width.

## The sync path, and why the fused router kernel is NOT worth building

The profile put "router + 48 syncs" at 15.0 ms/token (33%). Anatomy of one
invocation, median of 31, each step adding one more piece:

```
any GPU op + mx.eval (the floor)        ~171 us
qmm -> softmax                           174 us
  -> argpartition                        209 us
  -> take_along_axis (full router)       210 us
```

**~171 us of the 210 is the CPU-GPU round trip itself**, which no kernel can
remove. The router's whole four-dispatch chain is ~39 us. Collapsing it into one
fused kernel therefore caps out at 48 x 39 us = **1.9 ms, ~4% of a token** --
against real risk and a lot of work. **Do not build it.** This was item (2) on
the old kernel plan; it is now closed by measurement.

Cutting the *number* of syncs is the only real lever there, and it stays
architectural: routing for layer L+1 needs layer L's output, and the indices must
reach the CPU to drive reads. Escaping it means a GPU-side residency table so
`rhs_indices` can be computed on device — but the expert must be *provably*
resident before the GEMM runs, and prefetch recall is 87.6%, so that design
needs a correctness story before it needs a kernel.

**MLX streams do not help.** Hypothesis was that the router's `mx.eval` waits
behind the previous layer's queued expert GEMM, so putting the router on its own
stream would skip the wait. Measured: router+eval with a big GEMM queued costs
262 us on one stream, 269 us on two, and **286 us with nothing queued at all**.
It was never waiting on the GEMM. Hypothesis dead.

## Memory — searched for a free lunch, did not find one

At 3072 slots: 7.59 GiB pool + 0.83 GiB non-expert weights = 8.42 GiB active.
Process RSS is 7.80 GiB (untouched pool pages never fault in).

Three things tried, all negative — do not re-try them:

- **MLX's internal buffer cache holds 0.01 GiB.** `mx.clear_cache()` reclaims
  nothing. The memory is genuinely weights.
- **Over-fetching to trade spare bandwidth for capacity BACKFIRES.** The engine
  uses only 0.71 GB/s of the 3.5 GB/s available, so fetching the top (k x W)
  predicted experts instead of top k looked free. It is not: at 3072 slots,
  W=2 takes 22.56 -> 9.47 tok/s and true hit rate *falls* 96.6% -> 82.5%.
  Speculative fetches evict hot experts. **The pool is scarcer than the
  bandwidth.** Knob kept as `PF_WIDEN` (default 1) because the result is worth
  being able to reproduce.
- **Per-layer rebalancing has nothing to fix.** Slots resident per layer at
  steady state: mean 64.0, sd 7.3, cv 0.114, no layer starved or saturated. Only
  4.2% of the pool is misallocated versus perfectly even. LFU is already
  balancing the cyclic access pattern.

**So memory <-> speed is a genuine trade at 4 bits, and here is the curve**
(PF_DEPTH=1, cold, medians):

| slots | resident | tok/s | MB/token | true hit |
|---|---|---|---|---|
| 1536 | 4.60 GiB | 6.2 | 339 | 68.7% |
| 2048 | 5.87 GiB | 7.4 | 249 | 77.1% |
| 2560 | 7.14 GiB | 13.7 | 116 | 89.1% |
| **3072** | **8.40 GiB** | **22.6** | **34** | **96.6%** |

The only way to break the trade is **fewer bytes per expert**, which is the
quantization work -- and note it now pays on the GPU side too (the expert GEMM is
20.9 ms of weight reading at the 86 GB/s ceiling), not just on capacity. Doing it
without double-quantization damage needs the bf16 original, since everything on
disk is already 4-bit.

## Smaller fixes made

- **`np.unique` skipped at batch 1.** The top-k of a single token are distinct by
  construction, so unique is a no-op beyond sorting and `acquire` does not care
  about order. Prefill still uses the real path. Worth 0.16 ms/token (0.35%) --
  below the noise floor, kept because it is provably identical work removed, not
  because it is a speedup. Token ids verified bit-identical.

## Where the remaining time goes, and why it is stuck there

**The sync is GPU latency, not MLX overhead.** Splitting one router invocation
with `mx.async_eval` (queues without blocking) from the blocking eval:

```
build graph only            2.0 us
+ async_eval (submit)      18.1 us     <- all the CPU-side cost
+ eval (wait for GPU)     300.5 us     <- 282 us of pure round trip
mx.compile'd equivalent   238.9 us     (1.04x -- not a lever)
```

Submit is 18 us; the wait is 282 us. At 48 layers that is ~0.9 ms of CPU and
~13.6 ms of waiting for the GPU to acknowledge a 262K-MAC matmul. Nothing in
MLX or in our graph is responsible, so nothing in MLX or our graph can fix it.
Only issuing fewer syncs can, and that stays architectural (routing for L+1
needs L's output, and indices must reach the CPU to drive reads).

**Prefill is at the SSD limit, and bulk reads do not help.** Prefill activates
most of a layer's 128 experts, and repack.py made a layer's experts contiguous,
so one 339 MB sequential read should beat 128 scattered 2.53 MB ones. Measured:

```
128 scattered reads, 4 threads    99 ms   3.42 GB/s
1 contiguous 339 MB read          99 ms   3.42 GB/s
```

Identical. Phase 0's "larger tiles are faster per byte" was measured at queue
depth 1 and does not survive queue depth 4 -- the drive is already saturated
either way. So time-to-first-token (2.3 s for a 14-token prompt, 5.6 s for 412)
is bounded by bytes, not by access pattern, and the only fix is fewer bytes.

**And fewer bits is NOT available from the weights on disk.** Requantizing
4-bit -> n-bit (dequantize to bf16, requantize) measured against the current
4-bit model, on real experts and real activations:

| bits | blob | weight rel err | **MLP output rel err** | cos sim |
|---|---|---|---|---|
| 4 | 2.65 MB | 0 | 0 | 1.00000 |
| 3 | 2.06 MB | 9.9e-2 | **2.6e-1** | 0.94368 |
| 2 | 1.47 MB | 2.3e-1 | **4.8e-1** | 0.80943 |

The engine tolerates rel 2.0e-3 between two correct implementations of a layer
(measured: fused kernel fp32 accumulate vs gather_qmm bf16). **3-bit-from-4-bit
is 130x that**, and cos 0.944 means the output vector the residual stream carries
is visibly rotated. Double quantization is as bad as the folklore says, and now
it is measured rather than assumed.

**Conclusion: every remaining lever needs the bf16 original.** Decode is
GPU-latency + weight-bandwidth bound, prefill is SSD-bandwidth bound, and both
only move if experts get smaller. Quantizing once from bf16 to 3-bit would give
~2.06 MB blobs -- more slots per GB, less GPU traffic in the expert GEMM, and
faster prefill -- but quantizing twice destroys the model. ~60 GB download,
105 GB free.

## Robustness — RE-VERIFIED on the fixed pool (`verify_quality.py`)

The original robustness pass ran on a model with half its scales wrong, so it
was redone. gpt-oss-120b, 160 tokens per prompt, 600 slots:

| prompt | tok/s | memory | pool at the end | output |
|---|---|---|---|---|
| English prose | 2.31 | 8.95 -> 8.96 GiB | 600/600, pending 0, pinned 4 | coherent, on topic |
| Python | 1.98 | 8.95 -> 8.96 GiB | 600/600, pending 0, pinned 4 | valid, correct merge |
| French | 2.88 | 8.95 -> 8.97 GiB | 600/600, pending 0, pinned 4 | fluent, accents intact |

No leak, no repetition collapse, no mojibake, pool returns to a clean steady
state every time.

**And the retracted claim is now true.** "Output is bit-identical between 1536
and 3072 slots" could not have held before the fix, since 3072 was past the
overflow limit and 1536 was not. Measured on the fixed pool: **48/48 tokens
identical**, at 6.59 tok/s (1536) versus 17.11 tok/s (3072) -- the capacity
cliff, with the output proven unchanged across it.

## Robustness (original pass, on the corrupted pool — superseded)

160-token generation plus prose / code / French / 412-token prompts:
- coherent throughout, no repetition collapse, no mojibake
- **no leak**: 8.42 -> 8.45 GiB across all runs, peak 8.60
- pool returns to a clean steady state every time: pending 0, pinned 8,
  3072/3072 resident
- prefill 6-11 tok/s on short prompts, 73 tok/s on a 412-token prompt

One caveat for honest reporting: **decode throughput drifts +28% to +96% from
the first third of a generation to the last** as the pool warms. Benchmarks here
all warm up first, so they report steady state; a cold session starts around
9-18 tok/s and converges to ~22. Quoting 21.8 without that context overstates
what a first prompt feels like.

### Benchmark trap found the hard way

`ExpertPool.free` is a list popped from the END, so `acquire` fills slots
255, 254, ... A test that fills the pool and then reads slots 0..15 is reading
**zeros**, and comparing zeros to zeros looks like a perfect score. The first
run of the requantization test above reported 0.00e+00 error at every bit width
because of this. Always use the slot ids `acquire` returns.

## gpt-oss-120b — RUNNING. Architecture explored for speed; results below.

**It works.** 65.8 GB model, 8.97 GiB resident, coherent output (it correctly
enters gpt-oss's `analysis<|message|>` reasoning channel). Stock MLX cannot load
this model on this machine at all.

```
MOE_COLD=1 PF_SLOTS=600 PF_DEPTH=1 python3 engine_120b.py
  2.5 tok/s     1071 MB/token     true hit 53.4%     predict recall 84.5%
  600/4608 experts resident (13.0%)      prefill 33 s / 83 tokens
```

Store verified by `verify_120b.py` (216 experts sampled): all components
non-zero, nibbles spread, scales finite, index complete.

### Architectural levers, all measured

**1. top-k reduction is DEAD. The router is nearly flat.** Bytes/token =
layers x top_k x blob, and top_k is an inference-time knob, so halving it would
halve I/O. Measured over 864 real routing decisions:

| rank | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| mean softmax weight | 0.354 | 0.254 | 0.209 | **0.183** |

The 4th expert carries 18.3% of the mixture (25% would be perfectly uniform).
Truncating to top-3 discards a fifth of the output. Adaptive thresholding is no
better -- dropping experts below 20% of the top weight still reads 3.95 of 4,
and below 5% reads all 4. This is the load-balancing loss doing the same thing
it did to Qwen3's expert usage. **Do not revisit.**

**2. The SSD is capped at 3.41 GB/s regardless of read size.** Phase 0's
"larger tiles are faster per byte" (2.53 MB -> 2215 MB/s, 26 MB -> 4929 MB/s)
was measured at queue depth 1 and does not survive concurrency. At 14.02 MB
blobs: 1 thread 3.18, 2 threads 3.41, 4 threads 3.38, 12 threads 3.38 GB/s --
same ceiling Qwen3's 2.53 MB blobs hit. Read size is not a lever.

**3. Prefetch depth 1 is optimal; deeper pollutes the cache.** Same result as
Qwen3, and worse here because capacity is scarcer:

| config | tok/s | MB/token | true hit |
|---|---|---|---|
| **depth 1, 4 workers** | **2.51** | **1071** | **53.4%** |
| depth 1, 2 workers | 1.97 | 1071 | 53.4% |
| depth 2, 4 workers | 1.82 | 1401 | 42.6% |
| depth 1+2, 4 workers | 1.73 | 1524 | 40.3% |
| depth 1+2, 8 workers | 1.75 | 1524 | 40.3% |

4 workers beats 2 even though the raw SSD peaks at 2 threads -- the engine needs
the concurrency to overlap demand reads with prefetch.

**4. Capacity is the only thing that moves it, and it is capped by Metal.**

| slots | % of model | tok/s | MB/token | true hit | resident |
|---|---|---|---|---|---|
| 360 | 7.8% | 1.34 | 1574 | 31.5% | 5.83 GiB |
| 480 | 10.4% | 1.84 | 1380 | 38.9% | 7.40 GiB |
| **600** | **13.0%** | **2.51** | **1071** | **53.4%** | **8.97 GiB** |

**Measured after the fix: 700 slots is worth +5%, not the +16% extrapolated.**
Interleaved, order alternated, 4 rounds each, cold: 600 slots median 2.28 tok/s
(2.25-2.32), 700 slots median 2.40 (2.32-2.43), cache hit 37% -> 41%. The ranges
touch, so on this project's own standard that is not a throughput result by
itself -- but 700 won all four pairs and the hit-rate improvement is
near-deterministic, so the mechanism agrees. The +16% extrapolation came from a
capacity curve measured on the corrupted engine; the corrected curve is flatter.

The cost is headroom: 700 slots is 10.25 GiB resident against a 10.67 GiB
working set, which leaves almost nothing for a long KV cache. 600 remains the
safe default; use 700 for short contexts.

~~Still climbing steeply, but **613 slots is the hard ceiling**~~ — **no longer
true, and it was never the binding constraint it looked like.** The 8 GiB
`max_buffer_length` applies per buffer, and the pool is now twelve arrays rather
than one, so RAM is the cap: ~720 slots (working set 10.67 GiB). 700 slots
allocate and verify clean today, worth ~+16% by extrapolation from the curve
above. This also removes the argument that `kernels.py` was needed here to bind
several buffers.

Every row of the table above was measured on the corrupted pool. Bytes and time
do not depend on the scales, so the numbers stand; the *output* those runs
produced does not.

### Where the token goes at 2.51 tok/s

1071 MB/token at the achieved 2.68 GB/s is ~400 ms, and the token is ~398 ms --
so the engine is essentially **pure I/O now**, with compute fully hidden behind
it. That is the opposite of the 30B, which became compute-bound once its pool
crossed the capacity cliff. Everything else (fused kernels, sync reduction,
Python) is invisible here until bytes/token comes down.

## gpt-oss-120b — build notes (repack, bugs, geometry)

`repack_gptoss.py` (streaming repack) + `engine_120b.py` (engine) +
`verify_120b.py` (structural check). Model: `mlx-community/gpt-oss-120b-4bit`,
65.8 GB, 13 shards. **mlx_lm already implements `gpt_oss`**, so no architecture
work.

**Geometry vs Qwen3-30B**

| | Qwen3-30B | gpt-oss-120b |
|---|---|---|
| layers x experts | 48 x 128 | 36 x 128 |
| top-k | 8 | 4 |
| hidden / intermediate | 2048 / 768 | 2880 / 2880 |
| components per expert | 9 | **12** (real `bias` per projection) |
| blob | 2.53 MB (162 pages) | **14.02 MB (856 pages)** |
| expert store | 15.2 GB | **64.6 GB** |
| resident at 8 GiB pool | 3072/6144 = **50%** | 613/4608 = **13.3%** |

That last row is the whole story. The capacity curve above says 50% residency
is what buys 97.4% hit rate and 21.8 tok/s; 13.3% is well down the cliff, so
expect **single-digit tok/s**. Metal's 8 GiB `max_buffer_length` caps the pool
at 613 slots and RAM caps it near 700 anyway, so sharding the pool across
several buffers would not rescue it. The result is *120B running at all on a
16 GB machine*, not a speed record.

**Model differences that the engine has to handle**
- Routing is top-k of the RAW logits, then softmax over just the k selected.
  Qwen3 softmaxes all 128 first. Same argmax, but speculation needs no softmax
  at all -- argpartition on logits *is* the routing.
- Every projection carries a real bias on top of the quantization biases:
  twelve components per blob, and one of them is 1-D, which `ExpertPool` had to
  learn (it assumed 2-D).
- Clamped SwiGLU: clip gate to <=7, clip linear to [-7,7], then
  `gate*sigmoid(1.702*gate) * (linear + 1)`. The +1 is real, not a typo.
- `top_k` must be written into the repack index. The pool's barrier heuristic
  defaults to Qwen3's 8, which gives `36*8*2*1.25 = 720 > 560 slots` and
  switches the per-layer barrier ON -- 36 dead GPU round trips per token.

**Three bugs the streaming repack hit, all worth remembering**
1. **10 of 36 layers have their expert tensors split across two shards**
   (2,5,8,11,16,19,22,25,28,33). Assembling a whole blob in memory and writing
   it once silently drops half of those layers, because the other shard was
   already deleted. Write each component independently to its own offset and
   track progress per `(layer, proj, part)`.
2. **`os.path.realpath()` after `os.remove()` strands the blob.**
   `hf_hub_download` returns a symlink into `blobs/`; unlink it first and
   realpath returns the dead link path, so the 5 GB payload is never freed.
   Leaked 20 GB over four shards before it was caught. Resolve, then unlink.
3. **Non-expert tensors accumulated in memory until the end** are lost on any
   interruption, and the restart then re-downloads shards whose expert data is
   already on disk purely to recover ~1 GB. Save a per-shard partial as you go.

**Disk is the binding constraint, not RAM.** Store 64.6 GB against ~90 GB free
after reclaiming the redundant Qwen3-4bit HF cache (16 GB -- it is the source
`model/experts.bin` was built from, and the engine never reads it). Streaming
keeps peak at store + one shard. Two more cached models (Qwen2.5-14B-4bit,
Qwen2.5-7B-8bit, 15.2 GB) were removed when the projection showed a dip to
2.6 GB free; both are re-downloadable and unrelated to this project.

**Verification without the source.** Shards are deleted as consumed, so
`verify_repack.py`'s byte-exact comparison is not available. `verify_120b.py`
checks what survives: every component non-zero, 4-bit nibbles spread across the
codebook (a torn write or wrong offset collapses that), scales finite and
sane, and the index's written-set complete. It correctly flags a partially
written split layer -- layer 8 showed `gate_proj` present with `up`/`down` still
zero, which is exactly bug (1)'s signature.

## Model-internal structure — weight space exhausted, activation space is the open lead

First look inside the weights rather than at the I/O path. Measured on the real
120b store, dequantized to fp32, layer 18, 12 experts.

**Weight space: three decisive negatives. Do not revisit.**

| idea | result |
|---|---|
| shared base across experts (`W_i = B + D_i`) | mean expert holds **8.5%** of energy, deltas 91.5%; top-8 of 12 expert directions explain only 82% |
| per-expert low rank | rank 512/2880 = **66%** energy and already **1.42x MORE** bytes than 4-bit; rank 256 = 47.6% for a 29% saving |
| dedupe near-duplicate experts | cosine between distinct experts **+0.002** — orthogonal |

4-bit quantization already extracted the redundancy; what remains is dense,
full-rank and mutually orthogonal. There is nothing to compress *as weights*.

**Activation space: MEASURED AND DEAD. The 98% was a rank artifact. Do not revisit.**

The idea: `W @ x` only needs `W` restricted to the subspace `x` actually visits.
Write `P = Q Q^T`; then `W P = (W Q) Q^T`, so per expert you store `W Q` of shape
`[2880, k]`, and since Q is a property of the activations rather than the expert,
all 128 experts in a layer share one Q. It is a good idea. It does not work on
this model, because the subspace is not small.

**Why the old number said otherwise.** "90% at k=200, 98% at k=320" was fitted on
420 samples and evaluated on those same samples. A rank-k subspace fitted on N
samples explains its own fit almost perfectly as k approaches N, so at k=320 of
N=420 that measurement was mostly reporting its own degrees of freedom. Rerun
here on layer 18, same protocol, it reproduces exactly — and then falls apart the
moment anything is held out:

```
420 samples, fit and evaluate on the same rows   94.2% at k=320   <- the old protocol
420 samples, evaluated on held-out rows          32.6% at k=320
16,384 samples, evaluated on held-out documents  47.5% at k=320
```

**The measurement that replaces it.** 24,576 samples per layer over 8 genres
(wikitext, MMLU stem/humanities, dolly, trivia, swe-bench, source code,
9-language prose), all 36 layers, captured through the real engine; basis fitted
on two windows per genre and evaluated on the third. `corpus.py`,
`capture_acts.py`, `fit_basis.py`, `subspace_q1.py`. Everything is local — no
download.

k does not saturate anywhere useful (held-out, layer 18; other layers within a
few points):

| retention | 90% | 95% | 98% |
|---|---|---|---|
| k needed, of 2880 | **2108** | 2467 | 2702 |

and the estimate was still climbing with sample count — 22.8% → 47.5% at k=320
as the fit grew from 1,024 to 16,384 rows — so even this is generous.

**Variance retention is only a proxy, so the real thing was measured too.**
Relative error of the expert block's output, real 4-bit experts read from
`experts.bin`, real routing, clamped SwiGLU, held-out activations
(`subspace_q1b.py`; harness validated — a full-rank basis gives 1e-6):

| layer | k=320 | k=576 | k=960 | k=1408 |
|---|---|---|---|---|
| 4 (worst) | 0.862 | 0.735 | 0.603 | 0.475 |
| 18 | 0.531 | 0.451 | 0.375 | 0.302 |
| 35 (best) | 0.232 | 0.185 | 0.151 | 0.118 |

Scale: two correct implementations of a layer disagree at **2.0e-3**, and 3-bit
requantization — which visibly destroys the model — is **2.6e-1**. k=320 is at or
past that on every layer.

**And it is not PCA's fault.** The best *any* rank-k shared factorization can do
is computable: with `G = sum_e W_e^T W_e` and `C = E[u u^T]`, minimizing
`sum_e ||W_e (I - M) u||^2` over rank-k M is reduced-rank regression, and the
residual is the discarded singular values of `G^(1/2) C^(1/2)`
(`subspace_floor.py`). That floor at k=320 is **0.42** (layers 4 and 18) and 0.28
(layer 35); at k=720, still 0.32 / 0.22. PCA lands within ~0.1 of the floor, so a
cleverer basis buys nothing that matters — the rank is the problem.

**End to end**, projecting the expert input at k=320 through the real engine
while the router keeps seeing full x (`subspace_q2.py`, 8 held-out windows,
teacher forced):

```
perplexity  16.2 -> 1967  (x122)      top-1 agreement 13.9%      KL 4.34 nats
```

**bf16 vs 4-bit factors is moot.** 4-bit factors add 0.0016 on top of a 0.531
projection error at layer 18 — two orders of magnitude below it. For the record
the size breakeven is `2*(5760k + 5760) = 9,342,720`, so bf16 factors undercut
today's gate+up only for **k < 809**, and 4-bit factors only for k < 2883. At the
k where quality would survive (k ≳ 2400) there is nothing left to save.

**The two variants the shared-basis floor does NOT cover — also measured, also
negative** (`subspace_conditional.py`). The floor above only closes schemes where
one rank-k map is shared by every expert in a layer. Two members of the family
escape it, and both were worth testing on their own merits:

**A. A per-expert basis, conditioned on routing.** The floor used the *marginal*
activation covariance, but routing partitions the activation space — an expert
only ever sees its own slice. Economics allow it, because Q is shared between
gate and up (same input): per expert you store Q, W_g Q and W_u Q, three
[2880, k] against two [2880, 2880], which at 4 bits breaks even at k=1922.

Conditioning is real. Against a marginal basis fit on the *same* number of
samples — equal degrees of freedom, so neither side can win on rank — the
conditional basis is clearly better at every k:

| layer | k=320, per-expert | marginal at same n | shared basis (16k rows) |
|---|---|---|---|
| 4 | 40.7% | 24.5% | 39.5% |
| 18 | 43.2% | 32.4% | 44.8% |
| 35 | 62.8% | 47.6% | 57.1% |

So routing does tighten the subspace, by 10-16 points. It still loses or ties
against simply fitting one shared basis on 16,384 rows, because a per-expert fit
only gets ~900 tokens (24,576 tokens x 4 / 128 experts). More capture would
close that gap — and it does not matter, for the reason below.

**B. An output-side basis on down_proj.** Independent of everything above:
expert outputs all land in the same residual stream, so if they share a subspace
then `W_down ~ Q_out (Q_out^T W_down)`, with Q_out shared per layer and
[k, 2880] stored per expert. Retention of the realized outputs, held out:

| layer | k=128 | k=320 | k=576 | k=960 |
|---|---|---|---|---|
| 4 | 50.8% | 60.5% | 67.9% | 75.5% |
| 18 | 50.5% | 59.9% | 67.3% | 75.0% |
| 35 | **92.8%** | 94.1% | 95.1% | 96.2% |

The last layer really is concentrated — 92.8% at k=128. But it plateaus almost
immediately (92.8 → 96.2 across a 7.5x increase in k), which is a hard floor
rather than a slow curve, and the middle layers are no better than the input
side.

**Why none of this can be rescued: the bar is ~99.5%, not ~90%.** Calibrating
the measured retention-to-error mapping at layer 18 (`err = 0.80 *
sqrt((1-R) * 0.843)`, three points, matches to two digits):

```
output error 0.26 (3-bit requant, already destroys the model)  needs R = 87.4%
output error 0.10                                              needs R = 98.1%
output error 0.05 (still 25x the engine's own 2.0e-3 floor)    needs R = 99.5%
```

The best number produced anywhere in this study, across shared / per-expert /
output-side bases and every layer, is **96.2%** — layer 35's output side at
k=960, which is a 0.195 relative error while saving 20% of the blob. Everything
else is 40-75%. The gap is not a factor a better basis or more samples closes;
it is an order of magnitude in the residual.

**What is left untested, and why it is not worth testing.** A union of
subspaces (cluster the activations, one basis per cluster, pick at runtime) is
the obvious next variant, and the per-genre numbers say it would help on
quality. It cannot pay on size: the factors are per expert, so storing `W Q_c`
for c clusters multiplies per-expert storage by c, and the alternative —
selecting k of K columns from one large shared factor — turns each expert read
into a strided gather and destroys the contiguous-blob layout the whole store is
built on. Nonlinear encoders face the same 99.5% bar plus training and a custom
kernel. The genuinely open idea in this area is unrelated to bases: post-SwiGLU
`h` is contextually sparse, which is the "bonus finding" below.

**Third framing: score the error in KL instead of in activation energy
(the "J-space" question). Same answer.** `jspace_fisher.py`, `jspace_floor.py`.

Everything above prices a projection by how badly it reproduces `y`. Anthropic's
J-space result (July 2026) says the causally active part of the residual stream
is small -- ~25 verbalizable concepts, under 10% of activation variance -- found
via the input-output Jacobian rather than via variance. If that holds here, the
~99.5% bar was set in the wrong metric: error in a causally inert direction is
free, and a Jacobian-weighted basis would spend its rank only where the model's
predictions actually move.

That is testable. The right metric on the error is the Fisher at the expert-path
input (where a projection would act; the router is never projected):

    dKL ~ 1/2 E[du^T F du],   F = E[g g^T],   g = d NLL / d(expert input)

measured by backprop through the streaming engine with routing replayed from a
clean pass (argpartition's derivative is zero a.e.; scores stay differentiable;
`needs_barrier` forced on because the backward pass re-reads the slots). One
backward pass gives a gradient for every layer at once, 4 tokens at a time --
the sequence must be short enough that every expert it touches stays resident.

The second-order predictor calibrates: summed across layers it puts the PCA
basis at k=320 near 4 nats, against the 4.34 measured end to end by
`subspace_q2.py`. So it can be trusted to price alternatives.

And then the alternative does not survive being held out. Choosing the
Fisher-optimal basis on one half of the gradient samples and pricing it on the
other half (n=1000 samples for a 2880-dimensional metric):

| layer | PCA basis | Fisher-optimal, in-sample | Fisher-optimal, held out |
|---|---|---|---|
| 0 | 0.190 nat | 0.0004 nat | **0.260 nat** (worse than PCA) |
| 18 | 0.0126 nat | 0.0011 nat | 0.0119 nat (a tie) |
| 35 | 0.0316 nat | 0.0036 nat | 0.0248 nat (21% better) |

The 100x headroom in the in-sample column is **the same rank artifact for the
third time in this project** -- a Fisher estimated from 1000 samples has rank
1000 of 2880, so an "optimal" basis chosen against it discards every direction
the estimate never saw, exactly as the original k=320 basis discarded what 420
activation samples never saw. The J-space concentration itself does not replicate
either: Fisher energy in the top 25 directions is 93/50/49% in-sample and
1.8/15.9/39.6% held out.

**Honest limit of this one.** Unlike the `subspace_floor.py` result, which is
exact given C and G, this is a screening measurement bounded by sample count:
1000 gradient samples cannot determine a 2880-dimensional metric. Pushing it
properly wants >=10k samples, which at 4 samples per 2.5 s pass is ~2 hours.
What the screen does establish is that the Fisher-weighted basis is not *obviously*
better -- it ties or loses at two of three layers -- so the remaining upside is a
factor on a number that needs two orders of magnitude.

**A useful by-product: layers differ ~50x in how much they care.** Projecting
layer 27 away entirely costs 0.016 nats; layer 0 costs 0.446. Any future scheme
that spends bytes uniformly across layers is misallocating them, and this is the
cheapest signal yet for where to spend a fixed budget.

**Why the subspace is wide even though the spectrum looks steep.** Layer 18's
participation ratio is 277 of 2880 — a few directions really do carry enormous
energy — but the tail is thick: eigenvalue 1440 is still 0.5% of the top. Heavy
head, heavy tail. Per-genre bases are much tighter (71-75% at k=320 fitted on one
genre) and do not transfer (33-46% leave-one-genre-out), which is the same
stability wall the traffic-keyed bit allocation hit.

**Contextual sparsity of h — MEASURED AND DEAD, and it was never the layout.**
`sparsity_predict.py`. The note below says this is worth ~25% of bytes/token,
blocked on a transposed down_proj and a partial-read kernel. That figure counted
bytes and never priced the error, and the real obstacle was somewhere else
entirely: which rows of down_proj matter depends on `h`, which is not known until
gate/up have been read and multiplied, so the read is DEPENDENT and cannot be
issued a layer early like everything else in this engine. Thirty-six un-hidden
SSD round trips per token would eat the saving on their own.

There is one way out, and it is this project's own best trick one level down: at
layer L-1 the prefetcher already holds layer L's expert weights, so it can
compute a provisional `h` from layer L-1's hidden state, rank neurons by it, and
issue the row reads a full layer early. That works, partially -- predicted and
true row sets overlap 78-90% at useful sizes. It does not matter, because the
ORACLE does not pay either:

relative output error when only a fraction of down_proj's rows are read
(held-out tokens, error always computed with the true `h`, since the engine has
`h` and merely lacks the rows it did not read):

| rows read | blob | oracle | predicted a layer early | static set |
|---|---|---|---|---|
| 80% | 0.93x | 0.005-0.014 | 0.036-0.126 | 0.070-0.234 |
| 50% | 0.83x | 0.055-0.072 | 0.106-0.227 | 0.168-0.434 |
| 30% | 0.77x | 0.147-0.147 | 0.184-0.300 | 0.241-0.576 |

down_proj is a third of the blob, so reading fraction f of its rows takes the
blob only to (2+f)/3 -- half the rows buys **17%**, not 25%. At that point the
oracle already costs 5-7% output error and the schedulable version costs 11-23%.
Uniform 3-bit gets 22% of bytes for 26% error, and 3-bit is already known to
destroy this model, so contextual sparsity is strictly worse than an option
that was rejected. **Do not build the transposed store or the partial-read
kernel.** The 80.5%-of-energy-in-576-neurons statistic is true and irrelevant:
missing 20% of the energy is a 0.44 relative error.

Original note, kept because the sparsity statistic itself is real:

**Bonus finding, blocked by layout.** Post-SwiGLU h is contextually sparse:
90.2% of entries are below 5% of the row max, and the top 576/2880 hold 80.5%
of the energy. Skipping the rest would cut down_proj (a third of the blob). But
down_proj is row-major, so those contributions are strided sub-row reads --
useless for I/O. It needs down_proj stored **transposed** so each h-coordinate
maps to a contiguous row, plus a partial-read kernel. Worth up to ~25% of
bytes/token if someone wants it.

## Lower-level structure probes — three negatives and one small positive

Everything above attacks a 2880-wide *space*. These attack the blob by other
mechanisms entirely. `expert_neurons.py`, `weight_probes.py`.

**1. Static neuron pruning: no dead neurons at all.** Neuron j contributes
`h_j * W_down[:, j]`, so deleting it drops a row of gate and up AND a column of
down -- it scales the whole blob, which no subspace scheme can do, and needs no
kernel or layout change. Ranked by real contribution `E[h_j^2] *
||W_down[:,j]||^2`, selected on training tokens, scored on held-out tokens of
the same expert:

| keep | blob | held-out output error, L4 / L18 / L35 |
|---|---|---|
| 90% | 0.90x | 0.146 / 0.127 / 0.038 |
| 70% | 0.70x | 0.298 / 0.252 / 0.103 |
| 50% | 0.50x | 0.432 / 0.367 / 0.175 |

Neurons contributing under a millionth of the mean: **0.00% / 0.01% / 0.13%**.
The prior was good -- each expert sees ~1/32 of the tokens, so under-training
should leave dead units -- and it is simply false here. The intermediate width is
fully used. Error grows roughly linearly with the fraction removed, which is what
"every neuron matters a little" looks like.

**2. Gate/up pairing: real but far too weak.** A SwiGLU unit has two vectors,
`gate_j` and `up_j`. If `up_j ~ a_j * gate_j` you would store one vector plus a
scalar and halve gate+up, two thirds of the blob. Measured: mean |cos| between
paired rows is 0.118 against 0.021 for shuffled pairs, so the two halves of a
unit really are aligned about 5.6x above chance -- a nice mechanistic fact and
nothing more. The best per-neuron rescaling leaves **97.5%** of up's energy
unexplained.

**3. Permutation equivalence: tested properly, still negative -- but the old
argument was invalid.** HANDOFF previously concluded there is no cross-expert
redundancy because distinct experts have cosine +0.002. That reasoning does not
hold: cosine is not permutation invariant, and two experts computing the same
function with their hidden units in a different order would look exactly that
orthogonal. Neuron alignment is a real weight-symmetry phenomenon and the payoff
would be enormous (one expert plus a permutation is ~4 KB against 14 MB). So it
was actually tested -- Hungarian matching on the (gate, up) identity of all 2880
neurons between two experts:

```
best-matched neuron cosine   mean 0.083, max 0.393
||W_a - P W_b||^2/||W_a||^2  1.655   (2.0 = unrelated, 0 = permutation-identical)
```

Slightly better than unrelated, nowhere near equivalent. The conclusion survives;
the reasoning behind it has been replaced with a test.

**4. The one positive: the store is ~15% larger than its own information
content.** This is the only LOSSLESS axis and nobody had looked at it. Measured
over 16 experts at layer 18:

```
4-bit weight codes    3.669 bits of entropy per nibble (max 4.000)  -> 8.3% redundant
bf16 scales           high byte 1.56 bits, low byte 3.44 bits (max 8.00 each)
```

Weights are 89% of the blob, so entropy-coding them saves ~7.3% of it. The
scales are the softer target: they carry ~5 bits of real information in 16, and
scales+biases are 11% of the blob. **Storing scales as fp8 instead of bf16 is a
fixed-width, trivially decodable ~5.6% blob reduction** with no entropy coder
anywhere -- expand on read, which costs 1.5 MB of CPU work per expert against a
14 MB read. On a model that is purely I/O bound that is ~5.6% of throughput,
below the 12% noise floor to *measure* end to end but near-deterministic in
bytes/token, which is the number to check it on. Full entropy coding would
roughly double it and is much harder: variable-length decode at 3 GB/s is not
free, and it would have to happen on the GPU or in the reader threads.

## The bf16-original plan is impossible for gpt-oss-120b

Several places in this document conclude "every remaining lever needs the bf16
original". For the 120b that original **does not exist**: gpt-oss was released
*natively* in MXFP4 — the MoE projection weights were never distributed in
anything wider, and only the non-expert tensors are bf16
(https://huggingface.co/openai/gpt-oss-120b, model card arXiv 2508.10925). So
"quantize once, directly to the target width" is not available for this model at
any download size, and going to 3 or 2 bits from the MXFP4 release is exactly the
double quantization already measured as destructive (3-bit-from-4-bit: 2.6e-1
output error, cos 0.944).

Consequences for the plan:

- Uniform low precision, rotation-based quantization (QuaRot/SpinBhattacharya-
  style) and sensitivity-keyed mixed precision are all still open ideas, but on
  the **30B**, not the 120b -- Qwen3-30B does have a bf16 original at ~60 GB,
  against ~38 GB free disk today, so it needs disk reclaimed first.
- For the 120b specifically, the remaining byte levers are the lossless one
  above (~5.6% easy, ~15% hard), the contextual sparsity of `h` (~25%, needs a
  transposed down_proj and a partial-read kernel), and codebook quantization
  (AQLM/QuIP#-style vector quantization, which reaches ~2-3 bits at far better
  quality than affine and does not require a wider original -- but needs a
  custom kernel, and `kernels.py` is currently disabled).

## Where the engine is done / not done

| | status |
|---|---|
| Phase 0/0b — storage characterization | done |
| Phase 1/1b — routing telemetry | done |
| Phase 2 — baseline | done (resolved as: stock MLX *cannot* run it) |
| Zero-copy MLX extension | done, unplanned, works |
| Phase 3 — weight store | done, byte-verified |
| Phase 4 — expert cache | done (folded into slot pool) |
| Phase 5 — prefetcher | done |
| Phase 6 — Metal kernel | built + verified, at parity; now DISABLED, needs per-component bindings |
| Phase 7 — ship (README, repro bench, writeup) | done — `README.md`, `repro.py` |
| gpt-oss-120b (the actual headline) | running |
| Pool correctness (int32 overflow) | **found, fixed, verified** — `verify_pool.py` |
| Activation-subspace compression | measured, negative, closed |

The engine is functionally complete and correct. What remains is performance work,
packaging, and the bigger model.

## Next steps, in the order I'd take them

1. **Fix measurement.** Quiet machine, `bench.py`, re-baseline. Everything else depends
   on this.
2. ~~Fuse routing and speculation into one sync.~~ **DONE, +11%.** Both index sets
   now come back in a single `mx.eval`; 96 syncs per token dropped to 48. Concatenating
   the two gate *matrices* at load time on top of that measured 350 vs 386 us in
   isolation — a further ~10% of the router path, not yet done, easy.
3. ~~Move the pool hot path out of Python.~~ **Cancelled — measured at 0.4% of a
   token.** See the profile above. `acquire` is blocking on the SSD, not on Python.
4. **Mixed precision — but keyed on sensitivity, not traffic.** Bytes are still the
   binding constraint and `kernels.py` is still the only way to express per-expert
   bit widths. But the traffic-keyed version is measured and weakly supported: only
   ~28% of the hot/cold structure repeats across prompts (see "Expert traffic
   distribution"). Order of work: (a) uniform 3-bit baseline, which needs no
   stability assumption and gets 22% of bytes; (b) measure per-expert quantization
   sensitivity, which is prompt-independent by construction; (c) allocate on that.
5. ~~Hide more of the I/O by prefetching deeper.~~ **Tested, no win. Keep d=1.**
   See "Prefetch depth" below.
4. **gpt-oss-120b.** The engine is model-agnostic; this is mostly a repack. 62 GB
   download, 36 layers × 128 experts, top-4, 13 MB tiles (which Phase 0 measured
   *faster* than Qwen3's 2.53 MB), 1.83 GB/token vs Qwen3's 971 MB. This is what turns
   the project from "a nice optimization" into "120B on a 16 GB laptop."
5. **Phase 7.** README leading with the headline number, one-command reproducible
   benchmark, writeup on the LRU/LFU findings.

**Reordered after the pool bug (this is the current list):**

1. **Re-verify the quality claims on the fixed pool**, and re-check whether the
   capacity curve's *shape* survives — bytes and time do, but nothing was ever
   validated for correctness above the old limit. `verify_pool.py` is the guard;
   run it whenever the pool changes.
2. **Take the free capacity.** 700+ slots now work on the 120b (the 613 ceiling
   was a per-buffer limit). Worth ~+16% by extrapolation, and it is a one-line
   change to `PF_SLOTS` plus a memory check.
3. ~~Activation-subspace compression.~~ **Measured and dead.** See "Model-internal
   structure". Both weight space and activation space are now closed.
4. **Uniform lower precision from the bf16 original** is the only untried lever
   that is still standing, and it needs no stability assumption.
5. **Re-enable `MOE_FUSED`** by giving `kernels.py` per-component bindings, if
   the fused kernel is wanted back. It was at parity, so this is optional.

## On Metal kernels — RESOLVED: the expert block was never the problem

**This section previously said the workload was dispatch-bound and that fusing the
expert GEMMs was worth 20-30%. That was wrong, and the numbers behind it were
artifacts. It is corrected here; do not re-derive it.**

The claim was `single-token gather_qmm 297.9 us vs 92.1 us/token batched 8` →
"69% is fixed dispatch overhead". Two problems. Batching 8 tokens through the same
experts divides *weight bytes per token* by 8, so that comparison cannot separate
launch overhead from arithmetic intensity. And the 297.9 us was measured one
dispatch per `mx.eval`, so it is mostly the eval round trip, not the kernel.

Measured directly instead — 8 experts, cold distinct slots (every dispatch reads a
different set out of a 256-slot pool, so nothing is cached), R=32 dispatches per
eval to amortise the submit path, min of 9 **interleaved** trials:

```
empty kernel dispatch                               6.0 us   <- the real launch cost
read the 21.2 MB of expert blobs, no arithmetic   308.3 us
MLX gather_qmm path (3 gemms + swiglu + reduce)   319.9 us
fused single-dispatch expert kernel (kernels.py)  328.8 us
```

**The expert block at batch 1 is bandwidth-bound on weight bytes and MLX is already
within ~4% of the floor.** The entire MLP costs 12 us more than merely reading the
weights it multiplies. Seven dispatches per layer are ~42 us of launch overhead
against ~320 us of unavoidable traffic, so fusing them all away is worth ~1% of a
token. A fused kernel was built anyway (see below) and lands at parity, as it must.

Machine ceiling, for sizing anything else: peak GPU read is ~100 GB/s on a large
buffer; a 21 MB read reaches ~78-84 GB/s. Not a memory-compression artifact —
zeros, a repeated constant, and random bytes all measure within 4% of each other.
Threads per threadgroup matter more than threadgroup count: 8 TGs x 512 threads hit
96 GB/s where 8 x 256 got 52.

**Where the GPU-side time actually was: the router sync, not arithmetic.**

```
router matmul + top-k, sync amortised over 32 calls    20.2 us
router matmul + top-k, with its own mx.eval           209.8 us
```

~190 us of every router invocation is the CPU-GPU round trip. The engine took two
per layer — one to route, one to speculate — for 96 per token, despite both reading
the same hidden state and being ready at the same moment. **Fixed** (see below).

### What is done

1. **One sync per layer instead of two.** `streaming_call` now issues routing and
   speculation together and takes a single `mx.eval` for both. Speculation also
   dropped its softmax and `take_along_axis`: softmax is monotone, so the top k of
   the logits are the top k of the probabilities, and the prefetcher throws the
   scores away. Pure Python, no kernel. Token ids bit-identical before and after.
   **5.34 -> 5.92 tok/s (+11%)**, medians of 3 interleaved process pairs
   (prev 5.39/5.26/5.34, new 5.97/5.92/5.91); the new engine won every pair.

2. **Fused MoE expert kernel** — `kernels.py`, verified in `test_kernel.py`. One
   dispatch for the whole block: one threadgroup per active expert reads that
   expert's blob straight out of the slot pool as a flat uint32 buffer, computes
   gate and up in one pass over x in threadgroup memory, applies SwiGLU in
   registers, runs down_proj, and atomically accumulates the score-weighted result.
   The [8,768] intermediate never reaches device memory. It is at parity with MLX
   on speed (0.97-1.03x, and it cannot be better — see above) and **more accurate**:
   1.7e-4 vs MLX's 2.3e-3 relative error against an fp32 reference, because it
   accumulates in fp32 where MLX carries bf16 through the intermediate.
   **Now wired in** behind `MOE_FUSED=1` (default off), dispatching on shape so
   prefill still uses gather_qmm. End-to-end at 3072 slots, 3 interleaved rounds
   with order alternated: gather_qmm 22.79 tok/s, fused/nsg32 23.39 (1.03x),
   fused/nsg16 22.24 (0.98x) -- spreads overlap, so **still parity**, even now
   that compute is 78% of the token. Fusion is not the lever; bytes are.

   Its output differs from gather_qmm's, and that was checked rather than
   assumed: per-layer agreement is rel 2.0e-3, exactly the bf16 rounding floor,
   and under teacher forcing 12 of 13 steps pick the same top-1 token. The one
   flip is at a step where the top1-top2 logit gap was 0.125 -- the smallest
   representable bf16 gap -- against a 0.44 logit perturbation. Precision, not a
   bug, and the fused path is the more accurate of the two (fp32 accumulation).

   Kernel-side findings worth keeping: MLX's `metal_kernel` binds inputs as
   `const device T*`, not `constant`, so write `device` in the source. `bfloat16_t`
   is accepted as an input type. `atomic_outputs=True` + `init_value=0` gives
   `device atomic<float>*` and works. bf16 is readable off a uint32 binding with
   `as_type<float>(h << 16)`, which keeps the whole kernel on one buffer. Padding
   threadgroup memory to break the 32-stride bank conflict on x measured *slower*
   (the extra shift-add costs more than the conflict), so it is off by default —
   the loop is ALU-bound where it is not memory-bound.

### What is left, in the order the measurements support

1. **Per-expert mixed precision.** Now the highest-value item, not a curiosity.
   If the block is bandwidth-bound, bits removed convert roughly linearly into GPU
   time *and* into SSD traffic, and SSD traffic is ~70% of a token. `gather_qmm`
   takes one scalar `bits` per call so MLX cannot express it; `kernels.py` can.
   Measure the traffic distribution first — expert usage is near-uniform (Gini
   0.262), so "hot" is weak and the win may be smaller than it looks.
2. **Move the pool hot path out of Python.** `acquire` is now **69.6%** of accounted
   time. It is dict lookups, `np.unique` and lock traffic, and there is already a
   working C++ extension to put it in.
3. ~~Fused router + top-k kernel.~~ **Ruled out by measurement** -- ~171 us of the
   210 us router invocation is the CPU-GPU round trip, so fusing its four
   dispatches caps at ~4% of a token. See "The sync path" above.

Note the sync floor: 48 syncs/token are architectural, because routing decisions must
reach the CPU to drive reads. Escaping that means issuing prefetch from the GPU side,
which is a separate and much larger idea.

## Quantization / dequantization work — essentially untouched

The engine **consumes** existing 4-bit affine quantization (group_size 64) and never
reasons about it. Everything below is unstarted:

- **Per-expert mixed precision.** MLX's `gather_qmm` takes a single scalar `bits` for
  the whole call, so MLX structurally cannot express heterogeneous bit-widths across
  experts. Give hot experts more bits and cold experts fewer at the same average, and
  compare quality at equal bytes. Note the complication this project already found:
  expert usage is near-uniform, so "hot" is weak — the win may be smaller than it looks.
  Measure the traffic distribution on the target model first.
- **Bit allocation by measured traffic** rather than per-layer/per-channel. Novel axis;
  the instrumentation to gather the data already exists in `phase1_routing.py`.
- **Rotation-based quantization on Metal.** QuaRot/SpinQuant/ButterflyQuant are all
  CUDA-only algorithm papers. The field settles for cheap fixed Hadamard because online
  rotation costs compute — but at batch-1 decode on a bandwidth-bound machine the ALUs
  are idle, so that economics argument inverts. No Metal implementation exists.
- **2-bit KV cache** (RotateKV-style). At long context the KV cache is what actually
  kills you on 16 GB, and this engine currently ignores it entirely.

## LLM internals work — partially done

**Done:** expert usage distributions, cross-layer routing predictability, the finding
that adjacent-layer expert overlap is at chance while the hidden state predicts fine,
and that a trained probe beats the free projection by ~1 point. All in
`phase1_routing.py` / `phase1b_depth.py` with JSON results.

**Untouched, and interesting:**
- **Does quantizing an MoE change which experts fire?** Routing is a comparison between
  expert scores, and quantization perturbs those scores — so a quantized MoE may be
  routing tokens to different experts entirely, making the damage categorical rather
  than numerical. Nobody has looked. Cheap to check: log routing at fp16 vs 4-bit vs
  2-bit on identical inputs. This engine already has the routing instrumentation.
- **Expert co-activation structure.** Marginal usage is uniform, but that says nothing
  about correlation. If the co-activation graph partitions cleanly, packing co-activated
  experts into larger tiles buys up to 1.65× read throughput (measured ceiling). If
  activation is independent, packing is strictly harmful. Unmeasured, and it decides a
  real design question.
- **Massive activations / attention sinks** as a quantization-sensitivity story. The
  2026 literature converged on the outlier dimensions being a small, *input-independent*
  set — which means they can be identified offline and hard-coded, making mixed
  precision far easier to kernel.

## Working style this project has used

- **No fallbacks, no degraded paths, no simulators.** If something doesn't work, find a
  way that actually works. Several dead ends here were resolved by reading MLX's C++
  headers rather than accepting a Python-level limitation.
- **Never state a performance number that wasn't measured**, and never compare across
  runs without checking baseline stability first. This project has already had to
  retract one number for exactly that.
- **"Library X can't do Y" is not a stopping point** — it is the start of reverse
  engineering. The zero-copy extension exists because that assumption got challenged.

## Eviction — the LFU counter was never incremented, and fixing it is worth 16%

`engine_v3.py` declared `self.freq = defaultdict(int)` and read it in `_alloc`,
but **never wrote to it**. `engine_prefetch.py:228` still has the
`self.freq[key] += 1` that the v3 rewrite dropped. So every eviction candidate
scored 0 and `_alloc` degenerated to "skip pinned/pending/young, take the first
strided sample" — random eviction with a protection window. The LFU this file
and the findings above describe has never run on the 120b.

Restoring the counter (plus a `_touch()` on all three use paths: cache hit,
prefetch-hit wait, and demand miss) is behind `MOE_EVICT=lfu|lru|asis`, default
`lfu`. Interleaved, order rotated, 3 rounds, `MOE_COLD=1 PF_SLOTS=600
PF_DEPTH=1 TOKENS=96`. MB/token is near-deterministic; ranges nowhere near
touching on either metric:

| policy | MB/token | tok/s | cache hit | prefetch hit |
|---|---|---|---|---|
| asis (shipped) | 1400 (1400-1400) | 1.93 (1.92-1.95) | 41% | 46% |
| lru | 1279 (1275-1280) | 2.11 (2.10-2.14) | 49% | 39% |
| **lfu** | **1216 (1212-1222)** | **2.24 (2.24-2.25)** | 52% | 35% |

**1.15x fewer bytes, 1.16x throughput.** Decoded text is identical across all
three arms, as it must be — eviction changes only what is cached.

**Why frequency beats recency even though usage is near-uniform.** A prefetch
hit still performs a full read, so bytes track pool *admissions*, not demand hit
rate. Retention means the prefetcher finds an expert already resident and issues
no read at all: prefetch hits fall 46% -> 35% while cache hits rise 41% -> 52%.
Recency churns that working set; frequency holds it. The original "use LFU, not
LRU" guidance was right — it just was not implemented.

**Do not evaluate a cache policy for this engine with a demand-paging
simulator.** One was built here and it ranked LRU *above* LFU (59.4% vs 53.3%
on replayed routing), the opposite of the engine. It scored only the demand
stream, which is the half that does not dominate the byte count. A
prefetch-aware simulator (replaying the real free-projection predictions,
79.3% recall vs the engine's 81.8%) gets the direction right.

### Structure of the access trace — measured, and it bounds what is left

- **Consecutive tokens reuse 1.21 of 4 experts at the same layer, 9.7x chance**
  (0.125). Union over k consecutive tokens is sublinear: 8 tokens touch 17.9
  distinct experts per layer, not the 28.7 independence predicts. Cross-*layer*
  overlap was already known to be at chance; cross-*token* overlap is not.
- **Belady (offline optimum) at 600 slots is 79.1% against LFU's ~53%.** That
  bounds every policy. Reuse distance: median 4 tokens, only 19.1% within one
  token, p90 at 52 tokens.
- **Bounded-horizon Belady** — how far ahead a policy must see to capture it:
  1 token 62.7%, 2 tokens 64.8%, 4 tokens 68.2%, 8 tokens 72.9%, 16 tokens
  78.5%, infinite 80.4%.
- **A learned policy (LightGBM regressing log time-to-next-use, LRB/Parrot-style
  Belady imitation) reaches 68.3%** on held-out genres — trained on wiki /
  mmlu_stem / mmlu_hum / dolly / qanta, scored on swebench / code / multiling.
  Held-out R^2 0.49; gain is 83% `age`, so it is a well-calibrated LRU. This is
  a demand-paging number and has NOT been validated on the engine.

### Routing prediction cannot help eviction — closed by measurement

Predicting *which* experts future tokens want was the obvious way past the
learned policy. It is dominated:

- Same-layer **next-token** routing is only **28.5%** recall@4 from the hidden
  state, and the free projection adds nothing over persistence there (both
  0.285 at layer 18) — because applying layer L's router to x_L(t) just
  reproduces token t's own routing.
- Cross-layer lookahead decays d=1 75.4%, d=2 67.1%, d=4 59.1%, d=8 39.8%,
  d=18 20.6%, d=35 2.3% (chance 3.1%).
- A **perfect** next-token oracle scores 62.7% (the 1-token horizon row), which
  is **worse than the trace-only GBDT's 68.3%**. Belady's advantage lives 4-16
  tokens out, and that requires knowing text the model has not generated.

So a routing predictor, hypernetwork or subnetwork buys nothing for eviction
that reuse-distance statistics do not already buy more cheaply. **Do not build
one for this purpose.** (Prefetch is a separate question — there the signal is
cross-layer within one token, where recall is 75-84%.)

### What is left on the caching axis

LFU is shipped. The gap to Belady is still ~1.5x in bytes. The learned policy is
the only untried thing with evidence behind it, and it needs a prefetch-aware
re-evaluation first, since the demand-paging ranking did not survive contact
with the engine. Beyond that the wall is capacity, not policy: holding 16 tokens
of reuse needs ~2300 slots (~32 GB), so on 16 GB the honest ceiling is ~4 tok/s.

## The I/O path — where the 120b's missing 1.4x actually is

Decode-only profile (`scratchpad/profile_io.py`, 96 tokens, 600 slots, LFU
restored). `ExpertPool` already tracked `busy_s` and queue depth; nothing ever
printed them.

```
wall              463 ms/token (2.16 tok/s)     bytes 1057 MB/token
device busy       86.9% duty cycle
device IDLE       13.1%                <- schedulable
rate while busy   2.75 GB/s            <- the real problem
mean queue depth  2.79
```

The drive is busy 87% of the time, so idle is the *smaller* loss. The bigger one
is that it runs at 2.7 GB/s while working, on a device that does 3.5-3.8.

**Cause: the 13-segment scatter-gather read.** Each 14 MB blob is preadv'd into
3 projections x (weight, scales, biases, bias) plus padding, because the pool is
one array per component. Same bytes, same offsets, only the destination layout
differs (fresh offsets per run, 3 rounds, order alternated):

| layout | 1 thr | 2 thr | 4 thr | 8 thr |
|---|---|---|---|---|
| 1 segment (contiguous) | 3.46 | 3.82 | 3.77 | 3.84 |
| 13 segments (engine) | 2.23 | 2.92 | 3.39 | 3.59 |
| ratio | 0.64x | 0.76x | 0.90x | 0.93x |

Scatter costs 36% at queue depth 1 and ~10% at 4. The engine sits at depth ~2.8,
in the expensive part of the curve.

**BENCHMARK TRAP, again.** The first version of this reused the same 192 offsets
across all four layouts in a fixed order and reported the 13-segment layout at
5.79 GB/s -- *above* the contiguous ceiling, i.e. not physical. Later layouts
were reading blobs the earlier ones had warmed. Always draw fresh offsets per
run here; F_NOCACHE is not sufficient.

**PF_WORKERS is still not a lever -- retested and negative.** The table above
suggested 4->8 workers was worth ~6% with the real segment layout. On the engine
it is not: doubling workers raises mean queue depth 2.80 -> 3.42 but rate while
busy is unchanged (2.63-2.75 vs 2.72-2.76 GB/s, overlapping) and duty cycle gets
slightly worse (88% -> 87%). 12 workers is clearly worse (1.91-1.99 tok/s,
non-overlapping). There is not enough queued work to keep 8 threads fed --
reads are issued in per-layer bursts. The old "keep 4 workers" conclusion holds.

**DONE: staging buffer, `MOE_STAGE=1` (default on), +8.9%.** Read each blob
contiguously into a per-worker 14 MB buffer, then memcpy into the 13 segments.
Interleaved, order alternated, 2 rounds, via `profile_io.py`:

| | tok/s | rate while busy | MB/token |
|---|---|---|---|
| MOE_STAGE=0 | 2.27, 2.22 | 2.86, 2.78 GB/s | 1056, 1052 |
| **MOE_STAGE=1** | **2.45, 2.44** | **3.04, 3.05 GB/s** | 1062, 1052 |

80% -> 88% of the device ceiling, bytes unchanged, ranges non-overlapping. Pool
contents verified byte-identical to the scatter path. Costs 56 MB (4 workers x
14 MB). The isolated benchmark predicted 1.16-1.27x; the engine gives 1.09x,
the difference being contention the benchmark did not model.

**TRAP THAT COST AN HOUR: the pool segments are ctypes arrays, not
memoryviews.** They satisfy the buffer protocol so `preadv` scatters into them
correctly, but `arr[:] = src` invokes ctypes' per-element slice assignment:
**0.09 GB/s against 48.83 GB/s** through `memoryview(arr).cast('B')`. The first
version of this change ran at 0.07 tok/s -- 14.7 s/token, 35x slower than
baseline -- with completely correct output. `self.slots_mv` caches the byte
views once at construction. Anything that writes into the pool from Python must
go through those, never through `self.slots` directly.

**What is left:**

1. **The ~12% idle is mostly structural, not schedulable.** `prefetch(L+1)` is
   already issued before `acquire(L)` blocks, so speculation is in flight during
   the demand wait. There are only ~2.2 blob reads of work per layer, and L+1's
   prefetch cannot be issued before layer L's router runs, which needs L-1's
   output. Filling the gap means looking deeper, which is PF_DEPTH=2, already
   measured as a loss (recall 75% -> 67%, bytes 1071 -> 1401). Do not re-try.
2. ~~**The residual rate gap.** ... most likely GIL contention. Unmeasured.~~
   **MEASURED. It is not the GIL, it is burst structure.** See "The residual
   read-rate gap" below.
3. A true contiguous per-slot layout (repack) would remove the memcpy entirely,
   but the memcpy is not the cost -- the scatter was. Low priority.

Note the residual: even at depth 2.8 the benchmark does 3.0-3.4 GB/s with the
engine's own segment layout, against the engine's 2.7. **Resolved below: it is
burstiness, not GIL/lock contention.**

## The residual read-rate gap — RESOLVED. Not the GIL; the queue drains every layer.

`scratchpad/gil_io.py`, `scratchpad/burst_io.py`. The engine reads at 3.05 GB/s
on a drive that does 3.4-3.8 contiguous. Two hypotheses were standing; both are
now dead, and the real cause is structural.

**1. It is not GIL contention with MLX dispatch.** The suspicion was specific
and reasonable: `MOE_STAGE=1` ends every blob read with a 14 MB
`d[:] = buf[o:o+ln]` on memoryviews, and CPython's memoryview slice assignment
does **not** release the GIL, so four workers plus the main thread's dispatch
would serialise on it. Measured by driving the **real** `ExpertPool._read`
against the real `experts.bin` with the real ctypes segments (rule 1 — a
bytearray destination once predicted 1.2x for a change that was 35x slower),
while the main thread either slept or looped on a router-sized matmul + `mx.eval`:

| threads | main idle | main spinning on MLX |
|---|---|---|
| 1 | 2.478 | 2.537 |
| 2 | 3.431 | 3.446 |
| 3 | 3.513 | 3.490 |
| 4 | **3.513** | **3.516** |

Zero effect at every thread count. Replacing the memoryview copy with
`ctypes.memmove` (which *does* drop the GIL) also measured at parity. The
earlier "it is not a GIL problem" finding survives the staging change.

**2. It is burst structure.** The engine does not keep a queue. Per layer it
submits ~2.1 blob reads and blocks on all of them, so every burst ends at depth
1 and the device drains between layers. Mean queue depth over busy time can read
2.79 while most of the bytes actually move at depth 1-2. Same pool, same
segments, fresh random offsets, BURST reads submitted-then-joined in a loop:

| burst | GB/s while busy | mean depth |
|---|---|---|
| saturated | **3.487** | 3.98 |
| 1 | 2.732 | 1.00 |
| 2 | 2.691 | 1.50 |
| 3 | 2.932 | 2.00 |
| 4 | 2.853 | 2.50 |
| 8 | **3.241** | 3.25 |

Burst ~2 predicts 2.7-2.9 GB/s; the engine measures 3.05, the small surplus
being `prefetch(L+1)` overlapping `acquire(L)`. **The only fix is more reads per
join point**, and PF_DEPTH=2 (the obvious way to get them) is already measured
as a loss because it raises bytes. Multi-token verification is the other way,
and it raises burst size without raising bytes — see below.

**Measurement note.** An early version of this matrix produced rates up to 7.9
GB/s, which is not physical for this drive. It was an artifact of the alternating
loop, not of any config: eleven consecutive runs of one config, including one
reading 14.3 GB, sit at 3.48-3.59 GB/s with sd under 1.5%. If a number above
~3.6 GB/s appears here, do not believe it.

## Multi-token verification — the fixed cost of a forward is the biggest single lever

`scratchpad/batch_cost.py`, `scratchpad/union_trace.py`, `spec.py`.

**The finding: a forward pass costs a large amount that does not depend on how
many tokens are in it.** Measured by REPLAYING a fixed 64-token sequence so
every arm routes to the same experts in the same order and only the batching
differs (3 rounds, order alternated):

| t | ms/forward | ms/position | speedup | MB/position |
|---|---|---|---|---|
| 1 | 40.9 | 40.91 | 1.00x | 16.3 |
| 2 | 55.0 | 27.51 | **1.49x** | 15.4 |
| 4 | 85.7 | 21.42 | **1.91x** | 15.5 |
| 8 | 151.7 | 18.96 | **2.16x** | 15.6 |

Ranges are tight and non-overlapping. This fits **T(t) = 25.1 + 15.8t ms**, and
the 25.1 ms intercept is exactly the batch-independent part this document
already measured separately: 15.0 ms of 48 CPU-GPU sync round trips plus 9.3 ms
of attention/norms/lm_head, which are weight-bandwidth bound and so flat at
these batch sizes. Verifying t tokens in one forward pays that floor once
instead of t times. **This is the same 25.1 ms that "the sync path" section
concluded was architectural and unfixable — it is unfixable per forward, but it
is amortisable per token, and nobody had tried that.**

**The expert path amortises too, for an independent reason.** The union of
routed experts grows far slower than t, because consecutive tokens reuse experts
well above chance (`union_trace.py`, 384 decode tokens for the 30B / 256 for the
120b, 4 varied prompts, counted through the real pool):

| t | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 30b union/layer | 8.00 | 12.59 | 19.51 | 29.09 | 40.67 |
| 30b if independent | 8.00 | 15.50 | 29.12 | 51.62 | 82.42 |
| 120b union/layer | 4.00 | 6.84 | 11.42 | **18.11** | 26.45 |
| 120b if independent | 4.00 | 7.88 | 15.27 | 28.71 | 50.98 |

Correlation is worth 1.78x (30b) and 1.59x (120b) at t=8 against the
independent-routing model (`N(t) = E(1-(1-K/E)^t)`, MoESD arXiv 2505.19645).
The 120b figure **reproduces this document's separately measured 17.9**, which
is a useful cross-check on both.

**The 120b does not get the benefit, and the reason is capacity.** Same
measurement on gpt-oss-120b at 600 slots:

| t | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| speedup | 1.00x | 1.02x | 1.13x | **1.19x** |
| MB/position | 708.9 | 704.7 | 715.0 | 664.0 |

A t=8 forward needs `36 * 18.11 = 652` distinct expert slots against a 600-slot
pool, so **the working set of a single forward exceeds the entire pool** and the
union saving is eaten by thrashing inside one forward. Bytes barely move
(0.937x) where the union predicts 0.566x. The 30B is fine — t=8 needs ~1780 of
3072. **So this lever is for the 30B; on the 120b it is capacity-blocked, like
everything else on that model.**

### `spec.py` — built, correct, and currently NOT a win. The drafter is why.

Greedy speculative decoding on top of the streaming engine: draft γ tokens,
verify γ+1 positions in one forward, accept the longest prefix where the target
model's own argmax agrees with the drafter, take the model's token at the first
disagreement as a free bonus, and trim the KV cache back over the rejected
positions (`mlx_lm.models.cache.trim_prompt_cache`, which exists and works).

**The ceiling, with a perfect drafter** (`scratchpad/oracle_spec.py` — the greedy
continuation is pre-generated and handed to the drafter, so acceptance is 100%
by construction; the same "measure the oracle first" move that closed contextual
sparsity):

| arm | tok/s | range | MB/token | ratio | bytes | tok/step |
|---|---|---|---|---|---|---|
| base | 13.17 | 12.89-13.84 | 152.3 | 1.00x | 1.00x | 1.00 |
| oracle γ=1 | 13.97 | 13.89-14.47 | 154.1 | 1.06x | 1.01x | 1.46 |
| oracle γ=3 | 15.49 | 14.95-15.92 | 156.6 | 1.18x | 1.03x | 1.90 |
| oracle γ=7 | **20.20** | 19.59-20.24 | 149.0 | **1.53x** | **0.98x** | 7.31 |

Re-run: γ=7 gives 1.45x (19.05-19.30 vs 12.43-13.62). So **the ceiling on this
workload is ~1.5x**, and a perfect drafter costs **no extra bytes** (0.98x) —
every byte of overhead in the real arms is wasted drafts. γ=11 and γ=15 measured
*worse*, but that is an artifact: they left the script early (see the bf16 note
below) and the oracle then stops proposing. γ=7 is the operating point.

**What the free drafter actually delivers.** `spec.py` ships prompt-lookup
(n-gram) drafting, because the drafter has to cost no forward pass — see the
economics note below. 5 rounds, order alternated, warm pool, 192 tokens:

| arm | tok/s | range | MB/token | ratio | bytes | tok/step | accept |
|---|---|---|---|---|---|---|---|
| base | 14.38 | 12.38-14.94 | 107.3 | 1.00x | 1.00x | 1.00 | — |
| n-gram γ=4 | 11.28 | 10.77-11.51 | 148.8 | **0.78x** | 1.39x | 1.24 | 44% |
| n-gram γ=6 | 12.59 | 12.00-12.95 | 131.6 | **0.88x** | 1.23x | 1.19 | 42% |

**A loss.** At 96 tokens on the same prompt it is 1.11-1.12x; at 192 tokens it is
0.78-0.88x. The drafter fires on too few steps (0.45-1.03 proposals per step)
and at too low acceptance, so tok/step lands at 1.19-1.80 against the 7.31 the
oracle reaches, while every rejected position still pays a full set of expert
reads. **Report this as a negative on the drafter, not on the mechanism.**

**One real sub-finding: refusing to draft is a lever.** With `min_n=1` the
n-gram drafter almost always finds *some* match and proposes near-noise. Raising
the minimum match width to 3:

| min_n | accept | bytes | tok/step |
|---|---|---|---|
| 2 | 62% | 1.18x | 2.03 |
| 3 | **78%** | **1.08x** | 1.80 |

Bytes are the scarce resource here, so a draft that will not be accepted is
strictly worse than no draft. `min_n=3` is the default.

**Why no drafter that runs the stack can work.** Any drafter that does a forward
pass through the target — self-speculation at reduced top-k, an early-exit head,
a smaller MoE — pays that same 25.1 ms fixed cost per drafted token, which is
most of what is being saved. A 0.6B dense draft model is ~8-12 ms/token on this
machine, so with T(t) = 25.1 + 15.8t: γ=3 at 75% acceptance costs
25.1 + 63.2 + 30 = 118.3 ms for 2.73 accepted tokens = 43.3 ms/token against a
40.9 ms baseline — **a loss before it starts**. γ=2 is break-even. This matches
the public result that speculative decoding is net-negative on A3B-shaped MoEs,
but for a different reason than on a resident-GPU setup. **Do not build a
draft-model arm.**

**bf16 tie-breaking: "lossless" holds by construction, not observationally.**
The accept rule is exact — a drafted token survives only where it equals the
target's own argmax. But a batched forward does not compute bit-identical logits
to t sequential forwards, and at a bf16 near-tie the argmax itself moves. The
engine on its own **is** deterministic (baseline vs baseline: 192/192 identical
ids, checked). Speculative vs baseline is 64/64 at γ≤2 and 61/192 at γ=4 — i.e.
they agree until one near-tie flips, after which the sequences separate for
good. This is the same phenomenon already documented for the fused kernel (a
top1-top2 gap of 0.125, the smallest representable in bf16, against a 0.44 logit
perturbation). Quote it as "exact accept rule, bf16-order-dependent argmax", not
as bit-identical output.

### What this leaves open, in order

1. **A drafter that is free AND fires every step.** This is now the only thing
   between the engine and a measured 1.5x. The oracle says the mechanism pays
   and costs no bytes; the n-gram says prompt statistics are not enough. The
   candidate with the right economics is a **Medusa-style head**: one
   `[2048,2048]` residual block per head (4.2M params, ~8 MB) feeding the frozen
   `lm_head`, run on the final hidden state, so it costs ~1-2 ms and **no extra
   pass through the 48 layers**. Two or three heads at 60-70% acceptance would
   put tok/step near 3 at a cost of ~4 ms/step. Training is self-distillation on
   the engine's own greedy output, and `jspace_fisher.py` already demonstrates
   backprop through the streaming engine, so the machinery exists.
2. **Raise the 120b pool past its single-forward working set.** At 700 slots
   (already known to allocate and verify clean) a t=4 forward needs 555 — inside
   the pool for the first time. The 120b's t≥4 numbers above were all taken at
   600 and are capacity-floored; they are worth re-taking at 700.
3. The burst finding gives an independent reason to want multi-token forwards on
   the 120b even at low acceptance: it moves the per-layer read burst from ~2 to
   ~11-18, which `burst_io.py` prices at 2.7 -> 3.2 GB/s.

## Disk has changed and it closes a lever

This document says "~105 GB free" and repeatedly concludes "every remaining
lever needs the bf16 original", noting Qwen3-30B has one at ~60 GB. **There are
now 40 GB free.** The bf16 original does not fit, so uniform low precision,
rotation-based quantization and sensitivity-keyed mixed precision are blocked on
disk for the 30B as well as being impossible for the 120b (which was never
released in anything wider than MXFP4). Reclaim disk before planning around any
of them.

## Cleanup, 2026-08-18

The repo was 89 GB with 43 GB free, and disk had already closed one lever (see
"Disk has changed"). Removed:

- **`acts/*.npy` and `acts/basis.npz` — 10.4 GB.** Raw MLP-input activation
  captures and the fitted bases from the activation-subspace investigation,
  which is closed-negative. Regenerable by `capture_acts.py` + `fit_basis.py`.
  **The eleven JSON files in `acts/` were kept** — those are the findings
  (retention curves, output-error tables, Fisher results, the floor), and they
  are 120 KB.
- **All 19 root `*.log` files.** Every conclusion in them is already in this
  document; several were duplicate reruns (`capture`/`capture2`,
  `fit_basis`/`fit2`, `q1b`/`q1b2`, `jspace_in`/`jspace_in2`), and `chain.log`
  was empty.
- **`engine.py`, `engine_prefetch.py`** — superseded engine versions, imported
  by nothing.
- **`make_preds.py`, `pool_sim.py`** — the demand-paging pool simulator this
  document already records as unused and as having ranked LRU above LFU, the
  opposite of the engine.
- **`__pycache__/` and the CMake/ninja build intermediates under `ext/build/`.**
  The built `mlx_zerocopy_ext.cpython-312-darwin.so` and the sources
  (`ext/src/zerocopy.cpp`, `ext/CMakeLists.txt`) are kept; the extension still
  imports.

Deleted code and logs are archived in `.cleanup-archive-20260818.tgz` (18 KB).
The 10.4 GB of activations were not archived — that would defeat the purpose,
and they are regenerable.

Disk went 43 GB -> 53 GB free. Both engines re-verified after the cleanup:
Qwen3-30B 15.0 tok/s cold-start in 8.42 GB, gpt-oss-120b 2.32 tok/s in 8.96 GB,
both producing coherent text.
