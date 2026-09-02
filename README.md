moe-stream
----------

moe-stream runs mixture-of-experts language models whose weights do not fit in
memory. It keeps the non-expert weights resident and streams expert weights off
the SSD per token into a fixed pool of GPU slots, so the resident footprint is
set by the pool size rather than by the model. On a 16 GB M4 MacBook it runs
gpt-oss-120b, 65.8 GB on disk, in 8.95 GB.

moe-stream is built on MLX and is Apple silicon only. Stock MLX cannot load
either of these models on this hardware: `mlx_lm.load` materializes every
expert before the first token.

### Documentation quick links

* [Requirements](#requirements)
* [Getting the weights](#getting-the-weights)
* [Usage](#usage)
* [Benchmarks](#benchmarks)
* [How it works](#how-it-works)
* [Limitations](#limitations)
* [HANDOFF.md](HANDOFF.md) — full measurement record, including the dead ends

### Requirements

Apple silicon, macOS, Python 3.10+, MLX 0.32. About 80 GB of free disk if you
build both expert stores.

### Getting the weights

The repacked expert stores are not in this repository. They are 16 GB
(`model/`) and 62 GB (`model-120b/`) of derived data, built once from the
public MLX checkpoints:

```
$ huggingface-cli download mlx-community/Qwen3-30B-A3B-4bit
$ python3 repack.py            # -> model/
$ python3 repack_gptoss.py     # -> model-120b/
```

`repack_gptoss.py` downloads its own shards and is resumable; finished
components are recorded in `model-120b/experts_index.json`. `repack.py` reads
an existing Qwen3 snapshot from `~/.cache/huggingface/hub`, which is why it is
preceded by the download above.

Both output paths are hardcoded near the top of each script.

Verify the stores before timing anything:

```
$ python3 verify_repack.py     # 30B store against the source safetensors
$ python3 verify_120b.py       # 120b store
$ python3 verify_pool.py       # the slot pool
```

### Usage

```
$ python3 repro.py             # verify both pools, then benchmark both models
$ python3 repro.py --quick     # ~4 minutes
```

`repro.py` refuses to time a pool that does not byte-verify first, and
downloads nothing.

### Benchmarks

Measured on an Apple M4 (base), 16 GB unified memory, 10-core GPU, MLX 0.32,
with the page cache bypassed so every run is cold. The machine was at load
average 2.3 during these runs. Run-to-run spread is about 15%, so the ranges
are more meaningful than the medians.

| model | on disk | resident | tok/s | read per token |
|---|---|---|---|---|
| Qwen3-30B-A3B-4bit | 16.0 GB | 8.40 GB | 19.0 (17.6–19.1) | 32 MB |
| gpt-oss-120b-4bit | 65.8 GB | 8.95 GB | 2.35 (2.33–2.35) | ~1.1 GB |

Slot pool capacity drives almost all of this. Hit rate goes 68.7, 76.4, 90.7,
97.4% across 1536, 2048, 2560 and 3072 slots, and bytes read per token fall
from 344 MB to 32 MB, which is the point where the engine stops being I/O
bound.

### How it works

Only the non-expert weights stay resident, 0.83 GB of them. Everything else is
fetched per token.

MLX stores experts stacked per layer, so one expert's twelve
components live in twelve separate tensors and fetching it natively costs
twelve reads. `repack.py` rewrites the model so an expert is one contiguous,
page-aligned blob.

Staging is zero-copy. The pool is allocated by MLX, so it is a real
`MTLBuffer` the GPU reads in place, and `os.preadv` writes SSD bytes directly
into it. MLX has no zero-copy import path, since `mx.from_dlpack` copies, so
the allocation is inverted: MLX allocates and moe-stream reads into its buffer.

Eviction is LFU. `gather_qmm`'s `rhs_indices` can address any slot, so one set
of views serves every layer, and caching and prefetching become the same
mechanism. LRU is unusable here: expert access is cyclic over layers 0..47,
which is exactly the pattern that makes LRU evict every entry immediately
before it is needed again.

Prefetching is free. Applying layer L+1's router to layer L's hidden state
predicts L+1's experts at 85-88% recall, with no training and no extra weights,
because the residual stream keeps router inputs similar across layers.

### Limitations

Apple silicon only, and 4-bit MLX checkpoints only.

gpt-oss-120b runs at 2.35 tok/s. That is a working demonstration, not a usable
chat speed; it reads about 1.1 GB per token and the SSD saturates at 3.4 GB/s.

The expert store is close to its information-theoretic floor, so there is very
little left to win by compressing it further. The 4-bit codes carry 3.669 bits
of real entropy, which caps lossless gains at roughly 8% of the weight bytes.
Storing quantization scales as fp8 instead of bf16 is worth about 5.6% and is
the one easy win left.

Activation-space compression does not work, and `HANDOFF.md` documents the
attempt in detail. Storing `W Q` per expert against a shared per-layer basis
looked like a 2x win until it was evaluated on held-out documents; the subspace
needs 2108 of 2880 dimensions to retain 90% of the variance, and at k=320
end-to-end perplexity goes from 16 to 1967.

Prefetch depth beyond 1 does not help, and neither does over-fetching predicted
experts, which lowers the hit rate from 96.6% to 82.5% because speculative
fetches evict hot entries.

### Layout

```
repack.py, repack_gptoss.py                       safetensors -> expert-contiguous store
verify_repack.py, verify_120b.py, verify_pool.py  correctness checks
engine_v3.py                                      Qwen3-30B engine, and the shared slot pool
engine_120b.py                                    gpt-oss-120b engine
kernels.py                                        fused expert kernel (disabled, see HANDOFF.md)
bench.py                                          interleaved multi-trial benchmark harness
repro.py                                          one-command reproduction
HANDOFF.md                                        measurement record and dead ends
```
