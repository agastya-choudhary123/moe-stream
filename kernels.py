#!/usr/bin/env python3
"""
Fused Metal kernels for the streaming MoE engine.

The expert block at batch 1 is bandwidth-bound on weight bytes, and MLX's
gather_qmm is already within ~4% of the floor. Measured on this machine, 8
experts, cold and distinct slots, interleaved trials:

    read the 21.2 MB of blobs, no arithmetic at all   308 us
    gather_qmm path (3 gemms + swiglu + reduce)       320 us
    this fused kernel, best config                    329 us
    empty kernel dispatch                             6.0 us

So the whole expert MLP costs 12 us more than merely reading its own weights,
and fusing seven dispatches into one removes at most ~36 us per layer against
~320 us of unavoidable traffic. This kernel is at parity with MLX, not faster,
and no amount of tuning changes that -- the bytes are the cost.

(The premise it was built on, that ~69% of a gather_qmm call is launch
overhead, came from comparing 297.9 us at batch 1 with 92.1 us/token at batch
8. Batching 8 tokens through the same experts also divides weight bytes per
token by 8, so that comparison cannot separate launch overhead from arithmetic
intensity. Measured directly, a dispatch costs 6.0 us.)

What the kernel is for is per-expert mixed precision, which MLX structurally
cannot express because gather_qmm takes one scalar `bits` for the whole call.
That matters far more here than fusion did: if the block is bandwidth-bound,
bits removed translate roughly linearly into both GPU time and SSD traffic,
and SSD traffic is ~70% of a token. Owning this kernel is the prerequisite.

It is also more accurate than the path it replaces -- it accumulates in fp32
where MLX carries bf16 through the intermediate (1.7e-4 vs 2.3e-3 relative
error against an fp32 reference).

`moe_expert_mlp` collapses the whole expert block into one dispatch. One
threadgroup per active expert reads that expert's blob straight out of the
slot pool -- no gather, no strided view, no intermediate tensor -- computes
gate and up in a single pass over x held in threadgroup memory, applies SwiGLU
in registers, runs down_proj, and atomically accumulates the score-weighted
result into the output. The [8, 768] intermediate never reaches device memory.

Reading the pool as one flat uint32 buffer is what makes this possible. The
repacked store puts one expert's nine components in one contiguous 2.53 MB
blob, so a slot index plus the constant offsets below addresses every weight,
scale and bias in the layer. bf16 values are pulled out of the uint32 words by
hand (`bf16_lo`/`bf16_hi`) rather than typed, which keeps the whole kernel on
a single buffer binding.
"""

import json
import os

import mlx.core as mx

# Blob layout, in uint32 words, from model/experts_index.json. Every component
# offset is divisible by 4 bytes, so the whole 2.53 MB blob addresses cleanly
# as uint32 even though six of the nine components are bf16.
LAYOUT_U32 = dict(
    blob=663552,
    gate_w=0, gate_s=196608, gate_b=208896,
    up_w=221184, up_s=417792, up_b=430080,
    down_w=442368, down_s=638976, down_b=651264,
)

HEADER = """
// bf16 -> float without a bf16 type: the value is the high half of the fp32
// bit pattern. Keeps every access on the single uint32 pool binding.
inline float bf16_at(const device uint* base, uint i) {
    uint w = base[i >> 1];
    uint h = (i & 1u) ? (w >> 16) : (w & 0xffffu);
    return as_type<float>(h << 16);
}

// Threadgroup memory is 32 banks of 4 B. Lane l of a simdgroup reads the 32
// activations that pair with its own uint4 of weights, i.e. address 32*l + q --
// a stride of exactly 32, so on a classically banked memory all 32 lanes land
// in one bank. Padding one word every 32 makes the stride 33, coprime with 32,
// which would spread them. Measured: padding is a small loss here, so it is off
// by default -- the extra shift-add per access costs more than the conflict,
// which is the signature of an ALU-bound loop rather than a memory-bound one.
{pad32}
{pad8}
"""

# NSG (simdgroups per threadgroup) is templated in so threadgroup size can be
# swept; every loop below strides by NSG rows so any value divides the work.
SOURCE = """
    const uint e   = threadgroup_position_in_grid.x;
    const uint lt  = thread_position_in_threadgroup.x;
    const uint sg  = simdgroup_index_in_threadgroup;
    const uint ln  = thread_index_in_simdgroup;

    const device uint* B = pool + slots[e] * {blob}u;

    threadgroup float xs[XI32({d_model})];   // token, fp32, bank-padded
    threadgroup float xg[{ng_model}];   // per-quant-group sums of x
    threadgroup float gs[XI8({d_ff})];       // gate half, then h, bank-padded
    threadgroup float us[{d_ff}];       // up half
    threadgroup float hg[{ng_ff}];      // per-quant-group sums of h

    for (uint i = lt; i < {d_model}u; i += {nthreads}u) xs[XI32(i)] = (float)x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Group sums of x. Dequant is w = scale*q + bias, so the bias term of a
    // row's dot product is bias_g * sum(x over group g) -- computed once here
    // instead of once per weight.
    for (uint g = sg; g < {ng_model}u; g += {nsg}u) {{
        float s = xs[XI32(g * 64u + ln)] + xs[XI32(g * 64u + ln + 32u)];
        s = simd_sum(s);
        if (ln == 0u) xg[g] = s;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- gate_proj and up_proj, one pass ---------------------------------
    // Rows 0..d_ff-1 are gate, d_ff..2*d_ff-1 are up. MLX runs these as two
    // gather_qmm calls that each stream x in again; here x is already in
    // threadgroup memory and both halves are produced by one loop.
    for (uint r = sg; r < {two_ff}u; r += {nsg}u) {{
        const bool isup = r >= {d_ff}u;
        const uint row  = isup ? r - {d_ff}u : r;
        const device uint* W = B + (isup ? {up_w}u : {gate_w}u) + row * {row_w_model}u;
        const device uint* S = B + (isup ? {up_s}u : {gate_s}u) + row * {row_sb_model}u;
        const device uint* BI = B + (isup ? {up_b}u : {gate_b}u) + row * {row_sb_model}u;

        float acc = 0.0f;
        // Lane l takes uint4 l and l+32 of the row: 32 lanes x 16 B = a fully
        // coalesced 512 B line per step, and the 4 words of a uint4 always sit
        // inside one 64-element quant group, so one scale covers all of them.
        for (uint c = 0; c < {u4_per_row}u / 32u; c++) {{
            const uint w4 = ln + c * 32u;
            const uint4 v = ((const device uint4*)W)[w4];
            const uint j0 = w4 * 4u;
            const float sc = bf16_at(S, j0 >> 3);
            float dot = 0.0f;
            for (uint t = 0; t < 4u; t++) {{
                const uint w = v[t];
                const uint b = (j0 + t) * 8u;
                for (uint q = 0; q < 8u; q++)
                    dot = fma(float((w >> (4u * q)) & 0xfu), xs[XI32(b + q)], dot);
            }}
            acc = fma(sc, dot, acc);
        }}
        // One lane per quant group, so each bias lands exactly once.
        if (ln < {ng_model}u) acc = fma(bf16_at(BI, ln), xg[ln], acc);
        acc = simd_sum(acc);
        if (ln == 0u) {{ if (isup) us[row] = acc; else gs[XI8(row)] = acc; }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- SwiGLU, in place, never written to device memory -----------------
    for (uint i = lt; i < {d_ff}u; i += {nthreads}u) {{
        const float g = gs[XI8(i)];
        gs[XI8(i)] = (g / (1.0f + metal::exp(-g))) * us[i];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint g = sg; g < {ng_ff}u; g += {nsg}u) {{
        float s = gs[XI8(g * 64u + ln)] + gs[XI8(g * 64u + ln + 32u)];
        s = simd_sum(s);
        if (ln == 0u) hg[g] = s;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- down_proj, weighted, accumulated across experts ------------------
    for (uint r = sg; r < {d_model}u; r += {nsg}u) {{
        const device uint* W = B + {down_w}u + r * {row_w_ff}u;
        const device uint* S = B + {down_s}u + r * {row_sb_ff}u;
        const device uint* BI = B + {down_b}u + r * {row_sb_ff}u;

        float acc = 0.0f;
        for (uint c = 0; c < {w_per_row_ff}u / 32u; c++) {{
            const uint j = ln + c * 32u;
            const uint w = W[j];
            const uint b = j * 8u;
            float dot = 0.0f;
            for (uint q = 0; q < 8u; q++)
                dot = fma(float((w >> (4u * q)) & 0xfu), gs[XI8(b + q)], dot);
            acc = fma(bf16_at(S, j >> 3), dot, acc);
        }}
        if (ln < {ng_ff}u) acc = fma(bf16_at(BI, ln), hg[ln], acc);
        acc = simd_sum(acc);
        if (ln == 0u)
            atomic_fetch_add_explicit(&out[r], score[e] * acc,
                                      memory_order_relaxed);
    }}
"""

_CACHE = {}


def build(d_model=2048, d_ff=768, group_size=64, nsg=32, pad=False,
          layout=LAYOUT_U32):
    """Compile (or return cached) the fused expert kernel for one geometry."""
    key = (d_model, d_ff, group_size, nsg, pad)
    if key in _CACHE:
        return _CACHE[key]
    if d_model % (group_size * 4) or d_ff % (group_size * 4):
        raise ValueError("dims must be a multiple of 4 quant groups")
    hdr = (HEADER
           .replace("{pad32}", "#define XI32(a) ((a) + ((a) >> 5))" if pad
                    else "#define XI32(a) (a)")
           .replace("{pad8}", "#define XI8(a)  ((a) + ((a) >> 3))" if pad
                    else "#define XI8(a)  (a)"))
    src = SOURCE.format(
        blob=layout["blob"],
        gate_w=layout["gate_w"], gate_s=layout["gate_s"], gate_b=layout["gate_b"],
        up_w=layout["up_w"], up_s=layout["up_s"], up_b=layout["up_b"],
        down_w=layout["down_w"], down_s=layout["down_s"], down_b=layout["down_b"],
        d_model=d_model, d_ff=d_ff, two_ff=2 * d_ff,
        ng_model=d_model // group_size, ng_ff=d_ff // group_size,
        row_w_model=d_model // 8,          # uint32 per weight row (4-bit)
        row_sb_model=d_model // group_size // 2,   # uint32 per bf16 scale row
        row_w_ff=d_ff // 8,
        row_sb_ff=d_ff // group_size // 2,
        u4_per_row=d_model // 32,          # uint4 per weight row
        w_per_row_ff=d_ff // 8,
        nsg=nsg, nthreads=nsg * 32,
    )
    k = mx.fast.metal_kernel(
        name=f"moe_expert_mlp_{d_model}_{d_ff}_{nsg}_{int(pad)}",
        input_names=["pool", "slots", "x", "score"],
        output_names=["out"],
        header=hdr,
        source=src,
        atomic_outputs=True,
    )
    _CACHE[key] = (k, nsg * 32)
    return _CACHE[key]


def moe_expert_mlp(pool_u32, slots, x, scores, d_model=2048, d_ff=768,
                   group_size=64, nsg=32, pad=False, layout=LAYOUT_U32,
                   verbose=False):
    """One dispatch for a whole MoE layer at batch 1.

    pool_u32 : flat uint32 view of the slot pool
    slots    : uint32 [E]   slot index of each active expert
    x        : bfloat16 [d_model]
    scores   : float32 [E]  router weights, already normalised
    returns  : float32 [d_model]
    """
    kernel, nthreads = build(d_model, d_ff, group_size, nsg, pad, layout)
    e = slots.shape[0]
    return kernel(
        inputs=[pool_u32, slots, x, scores],
        grid=(e * nthreads, 1, 1),
        threadgroup=(nthreads, 1, 1),
        output_shapes=[(d_model,)],
        output_dtypes=[mx.float32],
        init_value=0,
        verbose=verbose,
    )[0]


def layout_from_index(path):
    """Read the blob layout out of the repacked store's index."""
    idx = json.load(open(path))
    off = {c["proj"].replace("_proj", "") + "_" + c["part"][0]: c["offset"] // 4
           for c in idx["components"]}
    off["blob"] = idx["blob_bytes"] // 4
    return off


if __name__ == "__main__":
    p = os.path.expanduser("~/Desktop/moe-stream/model/experts_index.json")
    got = layout_from_index(p)
    assert got == LAYOUT_U32, (got, LAYOUT_U32)
    print("layout matches store:", got)
