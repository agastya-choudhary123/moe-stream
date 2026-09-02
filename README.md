# moe-stream

**Running mixture-of-experts models that do not fit in RAM, on a 16 GB MacBook.**

A 65.8 GB model on a 16 GB machine, in 8.95 GB resident, producing coherent
text. Stock MLX cannot load either of these models here at all — `mlx_lm.load`
materializes every expert before the first token.

| model | on disk | resident | tok/s | read per token |
|---|---|---|---|---|
| Qwen3-30B-A3B-4bit | 16.0 GB | 8.40 GB | **19.0** (17.6–19.1) | 32 MB |
| gpt-oss-120b-4bit | 65.8 GB | 8.95 GB | **2.35** (2.33–2.35) | ~1.1 GB |

Apple M4 (base), 16 GB unified memory, 10-core GPU, macOS, MLX 0.32. Measured
cold (page cache bypassed) on a machine at load average 2.3, which is not a
quiet machine; run-to-run spread here is ~15%, so the ranges matter more than
the medians. Reproduce with one command:

```
python3 repro.py             # verifies both pools, then benchmarks both models
python3 repro.py --quick     # ~4 minutes
```

`repro.py` runs against the repacked stores on disk and refuses to time a pool
that does not byte-verify first. It downloads nothing itself — building those
stores is a one-time step, below.

## Getting the weights

The repacked expert stores are not in this repo: they are 16 GB (`model/`) and
62 GB (`model-120b/`), and both are derived artifacts. Build them once from the
public MLX checkpoints.

```
python3 repack.py            # mlx-community/Qwen3-30B-A3B-4bit  -> model/
python3 repack_gptoss.py     # mlx-community/gpt-oss-120b-4bit   -> model-120b/
```

`repack_gptoss.py` pulls its shards through `huggingface_hub` and is
resumable — it records which components have been written to
`model-120b/experts_index.json` and picks up from there, which matters for a
62 GB store. `repack.py` reads an already-downloaded Qwen3 snapshot out of
`~/.cache/huggingface/hub`, so fetch that one first with `huggingface-cli
download mlx-community/Qwen3-30B-A3B-4bit`.

Both output paths are currently hardcoded to `~/Desktop/moe-stream/` at the top
of each script.

Then verify before trusting any timing:

```
python3 verify_repack.py     # 30B store against the source safetensors
python3 verify_120b.py       # 120b store
python3 verify_pool.py       # the slot pool itself (see the corruption note below)
```

## How it works

Only the non-expert weights stay resident (0.83 GB). Expert weights are pulled
off the SSD per token and cached in a fixed pool of slots.

1. **Repack.** MLX stores experts stacked per layer, so one expert's twelve
   components are spread across twelve tensors and fetching it natively means
   twelve reads. `repack.py` rewrites the model so one expert is one contiguous,
   page-aligned blob — one read, no seek penalty at this size.

2. **Zero-copy staging.** The pool is allocated by MLX, so it is a real
   `MTLBuffer` the GPU reads in place, and `os.preadv` writes SSD bytes straight
   into it. Nothing is memcpy'd and nothing is allocated per token. (MLX has no
   zero-copy import path — `mx.from_dlpack` copies — so the trick is to invert
   it: let MLX allocate, then read into its buffer.)

3. **Slot pool with LFU eviction.** `gather_qmm`'s `rhs_indices` can address any
   slot, so one set of views serves every layer, and caching and prefetching
   become the same mechanism.

4. **Free prefetch.** Applying layer L+1's router to layer L's hidden state
   predicts L+1's experts at ~85-88% recall, with no training and no extra
   weights — the residual stream keeps router inputs similar across layers. A
   wrong guess costs one slot that eviction reclaims.

## Findings worth more than the throughput

**LRU returns literally 0% hits below 20% capacity here.** Expert access is
cyclic — layer 0..47, repeat — which is LRU's pathological case: when the cycle
is longer than the cache, it evicts every entry exactly before it is needed
again. LFU is immune. But plain LFU has a cold-start trap for prefetching: a
speculative entry arrives with frequency 0 and becomes the next victim, so you
evict precisely what you just fetched. Protecting entries younger than a bounded
window took the miss rate from 51% to 9.5%.

**Capacity has a cliff, and one measurement short of it gives the opposite
answer.** An early sweep tested 512/1024/1536/2048 slots, saw 2048 fail to beat
1536, and concluded "plateau". The next two points are 2.1x and 2.9x: true hit
rate goes 68.7 → 76.4 → 90.7 → 97.4% across 1536/2048/2560/3072, and bytes per
token collapse 344 → 32 MB, which is where the engine stops being I/O-bound.

**The pool was silently corrupting half of itself.** MLX shape dimensions are
int32, and `mx.view(uint32 -> bfloat16)` doubles the element count without
checking — it returned an array of shape `-1069285376`. Every quantization scale
read from a slot past `2^31/(blob/2)` came from the wrong address: slot 307 of
600 on the 120b, slot 1619 of 3072 on the 30B. The 4-bit weights were fine and
only their scales were wrong, so nothing crashed and the output stayed plausible
enough to pass for months. It showed up as a perplexity of 1020 on ordinary
English prose, and as two identical runs disagreeing. The pool is now one array
per component — no bit-cast, no strides, nothing left to overflow — and
`verify_pool.py` checks it. Fixing it also removed the 613-slot ceiling, since
`max_buffer_length` is a per-buffer limit.

**Speculating harder does not help, in three separate ways.** Prefetch depth 2
reads 20% more bytes for exactly zero throughput. Over-fetching the top k×2
predicted experts *lowers* the hit rate from 96.6% to 82.5%, because speculative
fetches evict hot entries — the pool is scarcer than the bandwidth. And
truncating gpt-oss's top-4 router to top-3 discards a fifth of the mixture: the
load-balancing loss flattened the router, so there is no cheap expert to drop.

**What is not the bottleneck**, each measured and each having been a plan at some
point: the expert GEMM (within 4% of the memory-bandwidth floor — no kernel can
beat it), fusing the router's four dispatches (~171 µs of a 210 µs invocation is
the CPU–GPU round trip), moving the pool hot path to C++ (0.4% of a token), and
larger reads (the SSD saturates at 3.4 GB/s regardless of size once queue depth
is 2+).

**The expert store is close to its own information-theoretic floor.** The 4-bit
codes carry 3.669 bits of real entropy, so at most ~8% of the weight bytes are
recoverable losslessly. Every structural axis that could have given more has been
measured and is absent: rank, input subspace, output subspace, per-expert
conditioning, causal/Fisher weighting, dead neurons (0.01% of them), permutation
symmetry between experts, gate/up pairing, and contextual sparsity — which fails
even given an oracle for which rows to read. The one easy win left is storing
quantization scales as fp8 rather than bf16: they carry ~5 bits of information in
16, which is ~5.6% of the blob.

**Both weight-space and activation-space compression are closed.** 4-bit
quantization already took the weight redundancy: experts are dense, full-rank,
and mutually orthogonal (cosine +0.002). The activation-subspace idea — store
`W Q` per expert against a basis shared by all 128 experts in a layer — looked
like ~2x until it was measured against held-out documents. The subspace needs
2108 of 2880 dimensions for 90% of the variance; at k=320 the expert block's
output is wrong by 23–86% and end-to-end perplexity goes 16 → 1967. The earlier
encouraging number was fitted and evaluated on the same 420 samples. A
per-expert basis conditioned on routing, and an output-side basis on down_proj,
were tested separately and land in the same place: usable quality needs ~99.5%
of the activation energy retained, and the best result anywhere in the study is
96.2%.

## Layout

```
repack.py, repack_gptoss.py   safetensors shards -> expert-contiguous store
verify_repack.py, verify_120b.py, verify_pool.py   correctness checks
engine_v3.py                  Qwen3-30B engine + the slot pool both engines use
engine_120b.py                gpt-oss-120b engine
kernels.py                    fused expert kernel (disabled, see HANDOFF.md)
bench.py                      interleaved multi-trial benchmark harness
repro.py                      one-command reproduction
HANDOFF.md                    everything measured, including the dead ends
```

`HANDOFF.md` is the real document. It records what was tried and ruled out, with
numbers, including several claims this project had to retract.
