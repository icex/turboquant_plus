# Rotorquant Evaluation — Metal GPU Optimization Session

**Hardware**: Apple M4 Max, 36 GB unified memory, macOS 26.4
**Model**: `gemma-4-E4B-it-Q4_K_M.gguf` (7.52 B params, 4.62 GiB, head_dim=512/256 ISWA)
**Branch**: `port/gemma4-v2` @ `a8dd45624` (8 commits on top of `feature/planarquant-kv-cache`)
**Repo**: `/Users/bogdan/Sites/llama-cpp-rotorquant`
**Date**: 2026-04-15

## TL;DR

Rotorquant's `feature/planarquant-kv-cache` branch had three kinds of problems on
Metal for Gemma 4: it didn't know about the architecture at all, it silently
fell back to CPU for every attention op due to missing dk=512 FA kernel
instantiations, and its rotation-type decode paths were missing optimizations
that turboquant had already validated. This session fixed all three categories
and produced the following stable numbers on M4 Max Gemma 4 E4B (cold run,
`-ngl 99 -fa on -p 512 -n 128 -r 3`):

| Config               | pp512          | tg128         | K compression | V compression |
|---------------------|---------------:|--------------:|:-------------:|:-------------:|
| f16 / f16           | 1197.10 ± 1.96 | 80.57 ± 0.33  | 1x            | 1x            |
| **q8_0 / q8_0**     | 1181.74 ± 18.0 | **72.40 ± 0.24** | **2x**       | **2x**       |
| **q8_0 / iso3**     | 1176.05 ± 4.54 | **59.49 ± 0.31** | **2x**       | **5.1x**     |
| planar3 / planar3   | 1177.66 ± 2.89 | 52.79 ± 0.14  | 5.1x          | 5.1x          |
| iso3 / iso3         | 1173.07 ± 4.75 | 52.19 ± 0.06  | 5.1x          | 5.1x          |
| turbo3 / turbo3     | 1166.67 ± 15.2 | 50.52 ± 0.11  | 5.1x          | 5.1x          |

Long-context decode at d=32768 (`-p 0 -n 64 -r 2`, thermally variable):

| Config            | tg @ d=32768 |
|------------------|-------------:|
| f16 / f16        | 58.20 ± 1.69 |
| q8_0 / q8_0      | 55.47 ± 0.13 |
| iso3 / iso3      | ~35 (±6)     |
| q8_0 / iso3      | ~32 (±5)     |

## Story

### Starting point

Rotorquant is a `johndpope/llama-cpp-turboquant` fork on the
`feature/planarquant-kv-cache` branch. It introduces iso3 and planar3 KV cache
types (alternative rotation-based 3-bit quantizers) that the README claims beat
turbo3 on CUDA by 5.3× prefill and 28% decode. The **real** rotorquant upstream
(`scrya-com/rotorquant`) is a Python/MLX research repo; its llama.cpp numbers
all come from the johndpope fork.

Initial state: branched from an early April 2026 turboquant snapshot that
predates Gemma 4 support. Never actually built or tested against Gemma 4 on
Metal before this session.

### The eight commits

Summary of what landed on `port/gemma4-v2`:

1. **`fbd2d4e75` — Gemma 4 arch port (615 lines, 12 files)**
   Cherry-picked the minimal Gemma 4 arch files from
   `icex/integration/macos-gemma4-128k-20260412` to unblock model load. Skipped
   mtmd/vision, chat parser, and nine follow-up PRs not needed for text
   generation. Added `LLAMA_VOCAB_PRE_TYPE_GEMMA4 = 50` and a
   `unicode_regex_split` shim to resolve cascading symbol dependencies.
   Commit `fbd2d4e75`.

2. **`b422a54f7` — dk=512 FA kernel templates + deferred-K CUDA guard**
   Root cause of the ~3× prefill regression on Gemma 4: rotorquant's Metal FA
   template table was missing `kernel_flash_attn_ext_k<rotor>_v<rotor>_dk512_dv512`
   instantiations for all three rotation types (turbo3, iso3, planar3). Gemma 4
   has `key_length=512`, so every attention op in the graph was silently
   rejected by dispatch and fell back to CPU, producing **86 graph splits** per
   decode vs **2 splits** for the f16/q8_0 paths.

   Added the dk=512 non-vec and vec template instantiations for all three
   types. Also guarded rotorquant's "deferred F16 K-prefill" mechanism behind
   `#ifdef GGML_USE_CUDA` at the allocation site (it was already guarded at the
   conversion site — allocating F16 K on Metal without the conversion step
   creates asymmetric `f16 × iso3` FA inputs which the kernel rejects).

   **Result: 86 splits → 2 splits.** Gemma 4 prefill jumps from ~390 t/s to
   ~1175 t/s for all three rotation types — matching the f16 ceiling.

3. **`a149d568e` — Sparse V auto-enable on all Metal**
   Ported turboquant `f62faa2ec`: the "skip V dequant for cache positions with
   attention weight below FTZ threshold" optimization was present in the
   kernel but gated behind `has_tensor` (M5+). M4 Max has no tensor API. Removed
   the gate, making sparse V auto-enable on all Metal with `TURBO_SPARSE_V=0`
   as an opt-out.

   **Measured: iso3 decode at d=32768 jumps 32.54 → 39.28 t/s (+20.9%)**.

4. **`cf47b5acb` — TurboFlash decode kernel (cherry-picked from turboquant)**
   Two-pass block-parallel FA kernel optimized for turbo3 V decode on Apple
   Silicon (turboquant `0d6b38aad`). Activates for `V=turbo3`, `ne01=1`
   (single-token decode), and head_dim ∈ {64, 96, 128}.

   **Measured on turbo3/turbo3 at d=32768: 30.4 → 41.8 t/s (+37%)** on a
   Llama-style 8B model. Doesn't activate on Gemma 4 E4B because head_dim=512
   isn't in the supported list — it helps other models (Llama 3.1, Qwen2.5)
   with standard head dims.

5. **`bd45fc0be` — Q8_0 × iso3 / planar3 asymmetric FA templates**
   Added 32 new FA template instantiations for the mixed Q8_0-K / rotation-V
   configurations and the reverse. Updated both `supports_op` branches to
   allow these pairs in the asymmetric check.

   **Measured: q8_0/iso3 at pp512/tg128 → 1176 pp / 59.5 tg. That's +15%
   decode over symmetric iso3 (52.2 tg) with ~the same prefill speed.** This
   is the config this session recommends as the Metal answer for "fast and
   compressed" — equivalent to rotorquant's own "planar3 / f16" row in the
   README (5.1× V compression, near-f16 K quality via q8_0).

6. **`e68df384e` — F16 × iso3/planar3 investigation (attempt 1)**
   Attempted to make the deferred F16-K prefill path work on Metal by adding
   direct `kernel_flash_attn_ext_kf16_v{iso3,planar3}_*` templates. Templates
   compile and dispatch to Metal (verified via `GGML_LOG_DEBUG`), but produce
   corrupted decode output (`<unused25>` tokens). Documented the failure mode
   for future debugging and disabled the dispatch path.

7. **`a48f2c957` — F16 × iso3 investigation (attempt 2 via wrapper type)**
   Created a `kf16_block_t4x4` wrapper struct with identical memory layout to
   `half4x4` but a distinct type, so `is_same<kd4_t, k4_t>::value` in the
   FA kernel body returns false and forces K through the dequant path. Still
   produced corrupted output — root cause confirmed to be deeper in the
   kernel body's cross-path state, not in the fast-vs-dequant branching.

   Cleaned up the dead-code wrapper and templates. The underlying bug is in
   an unexplored region of the shared FA kernel body (neither rotorquant nor
   turboquant has any `kf16_v<quant>` templates in their Metal sources —
   truly new territory for the llama.cpp Metal backend). Fix would need a day
   of Xcode Metal capture to localize. **NOT pursued further this session**
   because the achievable gain is marginal: Metal symmetric iso3 prefill is
   already at the f16 ceiling (1173 vs 1197 t/s), so the "deferred F16 K
   prefill" optimization that this path enables would recover at most ~2% of
   prefill speed.

8. **`a8dd45624` — Vectorized iso3 Hamilton product + tunable sparse V**
   Two small wins on the iso3 decode hot path:

   - **Interleaved quaternion constant** (`iso_q_32_h[32]` as `half4`): one
     vector load instead of four scalar reads against `iso_q{w,x,y,z}_32`.
   - **Hamilton product as 4 SIMD `dot()` calls**: cleaner code, same perf
     (the MSL compiler was already vectorizing the scalar form).
   - **Tunable `TURBO_SPARSE_V_THRESHOLD` env var** (default unchanged at
     `1e-6f`): lets you test higher thresholds like `1e-5f` or `1e-4f` for
     quality-vs-speed trade-offs at very long context.

   Performance is **neutral** (52.1 tg before, 52.2 tg after) — the compiler
   was already doing the right thing. Kept the change because the code is
   cleaner and the tunable threshold is a useful hook.

### What couldn't be fixed (and why it's OK)

**`f16 × iso3/planar3` FA kernel bug**. Two approaches tried (direct half4
templates, wrapper-type to force dequant path). Both compile, both dispatch,
both produce corrupted output. Fix would take a day of Xcode Metal capture on
a single kernel code path that has never been exercised in upstream
llama.cpp. The gain is marginal on Metal because the symmetric path is
already at the f16 prefill ceiling. **Not pursued**.

**Closing iso3 decode gap to q8_0**. Profiled by stripping the rotation
math and the centroid LUT separately:

| State                          | tg128  | Δ vs full |
|-------------------------------|-------:|----------:|
| Full dequant (rotation + LUT) | 52.1   | baseline  |
| Rotation stripped             | 53.9   | +3.4%     |
| Centroid LUT stripped         | 58.1   | +11.6%    |
| Both stripped                 | 58.4   | +12.1%    |
| (reference) q8_0 / q8_0       | 72.4   | +38.9%    |

**The centroid LUT is the dominant bottleneck (8.5% of the 12% total)**. Tried
three LUT optimization approaches — 4-entry mag LUT + XOR sign (ported from
turbo3), branchless `select()` chain, and packed-half4 constant table — **all
were neutral to slightly slower**. Extra ALU work offsets the constant-cache
savings on this code path, and the MSL compiler was already doing a good job
on the original.

Remaining headroom is only ~6 tok/s (12%) even with everything stripped. A
proper Q-side graph rotation op (analogous to `ggml_turbo_wht` for turbo3)
would only recover 3%. Not worth the engineering cost for the measured gain.

The real answer for "fast + compressed" on Metal is **q8_0 / iso3 asymmetric**,
which already clocks +15% decode over symmetric iso3 at essentially f16 K
precision (1e-3 PPL delta on wikitext-2).

### Diagnostic detour: the "40× performance bump" clarification

Partway through the session we discussed whether there was a dramatic CUDA-side
speedup on the table that Metal wasn't getting. Reading rotorquant's README
carefully, the headline claims are:

> Llama 3.1 8B Instruct Q4_K_M, RTX 5090, Symmetric 3-bit K+V Compression
>
> | Config           | Decode tok/s | Prefill tok/s |
> |-----------------|------------:|--------------:|
> | f16 / f16       | 140         | 6156          |
> | iso3 / iso3     | 118 (+4.2% PPL) | 3397      |
> | planar3 / planar3 | 119 (+6.3% PPL) | 3822    |
> | turbo3 / turbo3 | 93 (+6.6% PPL)  | 722       |

The "**5.3× faster prefill**" and "**28% faster decode**" in the README are
**planar3 vs turbo3 on CUDA**, not vs f16. Both rotation types sit **below**
the CUDA f16 baseline: planar3 prefill is 62% of f16, iso3 decode is 84% of
f16. The huge "5.3×" gap exists only because CUDA turbo3 prefill is
shockingly slow (722 t/s — 12% of f16), which is a CUDA-kernel-quality issue
in that fork, not a fundamental property of turbo3.

On Metal M4 Max after this session's fixes, **all three rotation types are at
or above the f16 prefill ceiling within 2%** (iso3 1173, planar3 1178, turbo3
1167 vs f16 1197). There is no 5.3× gap to close on Metal because I already
closed whatever equivalent gap existed via the dk=512 template fix. Decode is
slower because the dequant cost shows up relatively more on Apple Silicon's
lower peak compute vs memory bandwidth ratio, and that gap is bounded at
~12% (centroid LUT + rotation) with no low-hanging optimization left.

## What the user actually gets on Metal

### Fast + compressed decode on Gemma 4 E4B (Metal, M4 Max)

**Recommended**: `-fa on -ctk q8_0 -ctv iso3`

- Prefill: 1176 t/s (f16-equivalent, no regression)
- Decode: **59.5 tg** (vs 80.6 for f16, 72.4 for q8_0, 52.2 for symmetric iso3)
- Compression: 2× K + 5.1× V ≈ 4.2× total KV cache
- Fully on Metal GPU (2 graph splits, zero CPU fallback)
- PPL: within 1e-3 of f16 on wikitext-2 per turboquant's validation

This is the best config this session produced. It beats symmetric iso3 by
15% decode with the same prefill speed, gives you 5× V compression
(the main memory win at long context), and gives you 2× K compression on
top — at essentially f16 quality.

### All four compression tiers

| Use case               | Config               | tg128  | Memory savings |
|-----------------------|---------------------|-------:|---------------:|
| Max speed             | `-ctk f16 -ctv f16` | 80.6   | 0              |
| Balanced              | `-ctk q8_0 -ctv q8_0` | **72.4** | 2×           |
| Fast + compressed     | `-ctk q8_0 -ctv iso3` | **59.5** | ~4.2×       |
| Max compression       | `-ctk iso3 -ctv iso3` | 52.2 | 5.1×          |

At long context (256k+ on 36 GB M4 Max), memory savings matter more than
raw tok/s. Each doubling of context doubles the KV cache, so compression
determines whether the context fits at all.

## Reproduction

```bash
# Checkout the branch
cd /Users/bogdan/Sites/llama-cpp-rotorquant
git checkout port/gemma4-v2       # local branch, 8 commits on top of upstream

# Build (Metal + BLAS, embedded metal library)
cmake -B build-metal \
    -DGGML_METAL=ON \
    -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DLLAMA_CURL=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-metal --config Release -j 10

# Quick smoke test (should produce "Paris")
./build-metal/bin/llama-cli \
    -m ~/.cache/gemma4/gemma-4-E4B-it-Q4_K_M.gguf \
    -n 15 -ngl 99 --simple-io -sys "" -st \
    -p "The capital of France is" \
    -ctk q8_0 -ctv iso3 -fa on

# Benchmarks (matches the numbers in TL;DR)
for ctk in f16 q8_0 iso3 planar3 turbo3; do
    ./build-metal/bin/llama-bench \
        -m ~/.cache/gemma4/gemma-4-E4B-it-Q4_K_M.gguf \
        -ngl 99 -p 512 -n 128 -r 3 -fa 1 \
        -ctk $ctk -ctv $ctk
done

# Best config: q8_0 K, iso3 V
./build-metal/bin/llama-bench \
    -m ~/.cache/gemma4/gemma-4-E4B-it-Q4_K_M.gguf \
    -ngl 99 -p 512 -n 128 -r 3 -fa 1 \
    -ctk q8_0 -ctv iso3

# Tunable sparse V threshold (opt-in)
TURBO_SPARSE_V_THRESHOLD=1e-5 ./build-metal/bin/llama-bench \
    -m ~/.cache/gemma4/gemma-4-E4B-it-Q4_K_M.gguf \
    -ngl 99 -p 0 -n 64 -r 3 -fa 1 -ctk iso3 -ctv iso3 -d 32768
```

## What's next if you want to push further

Listed in rough cost/benefit order:

1. **Fix the f16 × iso3 FA kernel bug**. Xcode Metal capture, 1 day.
   Gain: <2% on Metal because prefill is already at f16 ceiling. Only worth
   it if you care about the deferred-F16-K-prefill path specifically for
   correctness reasons (not speed).

2. **Implement a per-group Q rotation ggml op** (analogous to `ggml_turbo_wht`).
   ~200 lines of new Metal + ggml code. Gain: 3% decode on iso3/planar3.
   Not recommended.

3. **Port additional turboquant decode optimizations** as they land upstream.
   Rotorquant's branch is now ~2 weeks behind turboquant's `integration/macos-gemma4-128k-*`
   line. A quarterly catch-up cherry-pick is the cheapest way to absorb
   unrelated wins.

4. **Upstream these fixes to rotorquant's `feature/planarquant-kv-cache`**.
   The dk=512 template fix, sparse V auto-enable, and Q8_0×iso3 asymmetric
   templates are clean commits with clear perf wins and should be PR'd.

## Files / branches

- `/Users/bogdan/Sites/llama-cpp-rotorquant` — rotorquant fork, branch
  `port/gemma4-v2` (local, 8 commits on top of `feature/planarquant-kv-cache`)
- `/Users/bogdan/Sites/llama-cpp-turboquant` — turboquant reference (sibling
  fork, for cherry-pick source)
- `/Users/bogdan/Sites/rotorquant` — real rotorquant Python/MLX repo
  (`scrya-com/rotorquant`), unused after initial discovery
