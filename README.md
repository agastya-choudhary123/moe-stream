# moe-stream

Runs mixture-of-experts models that don't fit in memory on Apple silicon. The
non-expert weights stay loaded. Expert weights are read from the SSD as each
token needs them, into a fixed-size pool of GPU slots. Memory use is set by
the pool size, not the model size. On a 16 GB M4 MacBook, it runs
gpt-oss-120b (65.8 GB on disk) in 8.95 GB.

It's built on MLX. Stock `mlx_lm.load` can't run either model on this
hardware, because it loads every expert before generating the first token.

## Requirements

- Apple silicon, macOS, Python 3.10+, MLX 0.32
- About 80 GB of free disk if you build both expert stores

## Getting the weights

The repacked expert stores aren't in the repo, because they're 16 GB
(`model/`) and 62 GB (`model-120b/`). You build them from the public MLX
checkpoints:

```sh
huggingface-cli download mlx-community/Qwen3-30B-A3B-4bit
python3 repack.py            # -> model/       (reads the HF cache)
python3 repack_gptoss.py     # -> model-120b/  (downloads its own shards, resumable)
```

The output paths are hardcoded near the top of each script.

Check the stores before benchmarking:

```sh
python3 verify_repack.py     # 30B store vs. the source safetensors
python3 verify_120b.py       # 120B store
python3 verify_pool.py       # slot pool
```

## Running

```sh
python3 repro.py             # verify both pools, then benchmark both models
python3 repro.py --quick     # about 4 minutes
```

`repro.py` won't benchmark a pool unless it verifies byte-for-byte first, and
it doesn't download anything.

## Benchmarks

Apple M4 (base), 16 GB, 10-core GPU, MLX 0.32. The page cache was bypassed,
so every run is cold. Runs vary by about 15%, so the ranges matter more than
the medians.

| model | on disk | resident | tok/s | read per token |
|---|---|---|---|---|
| Qwen3-30B-A3B-4bit | 16.0 GB | 8.40 GB | 19.0 (17.6–19.1) | 32 MB |
| gpt-oss-120b-4bit | 65.8 GB | 8.95 GB | 2.35 (2.33–2.35) | ~1.1 GB |

Pool size is the main factor. For the 30B model, going from 1536 to 2048,
2560, and 3072 slots raises the hit rate from 68.7% to 76.4%, 90.7%, and
97.4%, and the data read per token drops from 344 MB to 32 MB. At that point
it's no longer limited by I/O.

## How it works

- **Contiguous experts.** MLX stores experts stacked per layer, so a single
  expert is spread across twelve tensors and takes twelve reads to load.
  `repack.py` rewrites the model so each expert is one contiguous,
  page-aligned blob.
- **Zero-copy loading.** MLX allocates the slot pool, so it's a real
  `MTLBuffer` that the GPU reads directly. `os.preadv` reads from the SSD
  straight into it. (`mx.from_dlpack` copies, which is why MLX has to do the
  allocation.)
- **LFU eviction.** `gather_qmm` can index any slot, so every layer shares
  one set of views. LRU doesn't work here: experts are accessed in a cycle
  over layers 0–47, and LRU would evict each entry just before it's needed
  again.
- **Prefetching.** Running layer L+1's router on layer L's hidden state
  predicts which experts L+1 will use with 85–88% recall. That works with
  no training and no extra weights, because the residual stream changes
  slowly between layers.

## Limitations

- Apple silicon and 4-bit MLX checkpoints only.
- 2.35 tok/s on gpt-oss-120b is too slow for chat. It reads about 1.1 GB
  per token, and the SSD maxes out at 3.4 GB/s.
- There's not much left to gain from compression. The 4-bit codes have 3.669
  bits of entropy, which limits lossless savings to about 8%. Storing the
  quantization scales as fp8 instead of bf16 would save about 5.6%.
- Compressing in activation space didn't work. A shared per-layer basis
  looked like a 2x saving until I tested it on held-out text: keeping 90% of
  the variance needs 2108 of 2880 dimensions, and at k=320 perplexity goes
  from 16 to 1967. HANDOFF.md has the details.
- Prefetching more than one layer ahead didn't help. Fetching extra predicted
  experts made things worse (hit rate fell from 96.6% to 82.5%), because the
  speculative fetches pushed out experts that were actually in use.

## Layout

```
repack.py, repack_gptoss.py   build the expert stores
verify_*.py                   correctness checks
engine_v3.py                  Qwen3-30B engine and the shared slot pool
engine_120b.py                gpt-oss-120b engine
kernels.py, test_kernel.py    fused expert kernel (disabled, see HANDOFF.md)
ext/                          C++ zero-copy read extension and its tests
bench.py, repro.py            benchmarks
HANDOFF.md                    full measurement log, including what didn't work
```

The other scripts in the root (`subspace_*.py`, `jspace_*.py`,
`capacity_sweep.py`, `codebook_quant.py`, and so on) are the experiments
described in HANDOFF.md, and their outputs are in `acts/`. `scratchpad/` has
one-off I/O and caching experiments.
