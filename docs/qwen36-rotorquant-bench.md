# Qwen 3.6 35B-A3B on rotorquant (MoE bench)

**Hardware**: Apple M4 Max, 36 GB unified memory, macOS 26.4
**Model**: `unsloth/Qwen3.6-35B-A3B-GGUF : Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (22.4 GB)
**Architecture**: `qwen35moe` (MoE — 35B total, 3B active per token, 256 experts × 8 active)
**Hyperparameters**: 40 layers, hidden 2048, 16 attn heads / 2 KV heads (GQA 8:1),
head_dim 128, context length 262144
**Binary**: `llama-cpp-rotorquant` @ `wip/rotorquant-metal-checkpoint-20260415` (`a48f2c957`)
**Flags**: `-ngl 99 -fa on -r 3`
**Date**: 2026-04-16

## Results

### Phase 1: pp512 + tg128 (decode from cold cache)

| Config                  | pp512 t/s     | tg128 t/s    | vs f16 decode |
|------------------------|--------------:|-------------:|:-------------:|
| **f16 / f16**          | 1164.20 ± 4.7 | **63.18**    | baseline      |
| **q8_0 / q8_0**        | 1167.78 ± 7.8 | **61.75**    | −2.3%         |
| q8_0 / iso3            | 1152.05 ± 4.5 | 58.32        | −7.7%         |
| planar3 / planar3      | 1160.25 ± 6.7 | 56.29        | −10.9%        |
| iso3 / iso3            | 1156.45 ± 8.3 | 55.74        | −11.8%        |
| turbo3 / turbo3 (TurboFlash) | 1135.11 ± 3.6 | 55.61  | −12.0%        |

### Phase 2: tg64 @ d=16384 (long-context decode)

| Config                  | tg64 @ 16k     | vs f16 decode |
|------------------------|---------------:|:-------------:|
| **f16 / f16**          | **51.59 ± 6.3**| baseline      |
| **turbo3 / turbo3** (TurboFlash) | **42.85 ± 3.7** | −16.9%    |
| q8_0 / q8_0            | 42.66 ± 4.3    | −17.3%        |
| iso3 / iso3            | 41.35 ± 4.2    | −19.9%        |
| q8_0 / iso3            | 38.55 ± 5.7    | −25.3%        |

Noise is high in Phase 2 (±4-6 tok/s) due to M4 Max thermal variability over
the 10-run sweep. Relative ordering is stable across reruns but absolute
differences within ±5 are noise.

## Key findings

### 1. Rotation-type KV compression is a **dense-model** optimization

On Gemma 4 E4B (7.5B dense), our champion config was `q8_0/iso3` asymmetric at
60 tg vs 52 for symmetric iso3 and 72 for q8_0 — the iso3 V cache saved real
bandwidth because the model is memory-bound per token.

On Qwen 3.6 35B-A3B (MoE, 3B active but all 35B weights touched via routing),
**the exact same config `q8_0/iso3` is the slowest tested**. The KV cache is
already tiny (GQA 8:1 × 2 KV heads × head_dim 128 = 2048 elements per token
per layer) and decode is dominated by weight bandwidth. Compressing KV further
only adds dequant overhead without proportional bandwidth savings.

**Rule of thumb**: rotation-type KV compression pays off when KV bandwidth is
≥20% of decode time. For dense models with large head dim / few KV heads
(Gemma 4 E4B: head_dim 512, 2 KV heads → ~16 KB KV/token/layer), it's worth
it. For MoE with tight GQA (Qwen 3.6: head_dim 128, 2 KV heads → ~512 bytes
KV/token/layer — **32× smaller relative to model weights**), it isn't.

### 2. TurboFlash finally proved itself

On Gemma 4 E4B (head_dim 512), TurboFlash never activated — dk=512 isn't in
its {64, 96, 128} supported head_dim set. This was one of the items the
session couldn't validate.

On Qwen 3.6 (head_dim 128, V=turbo3, ne01=1) all three TurboFlash activation
conditions are met. Result: **turbo3 decode at d=16k matches q8_0** (42.85 vs
42.66) — the first data point in the session where the TurboFlash port from
turboquant's `0d6b38aad` actually pays its way. **5.1× V compression at
matched decode speed** is meaningful for long-context workloads.

### 3. f16 is still the fastest decode on Qwen 3.6 at every depth

Unlike Gemma 4 where q8_0 decode is within 10% of f16 across the board, on
Qwen 3.6 f16 consistently wins. The Qwen 3.6 KV cache is small enough that
the dequant overhead for ANY quantized KV costs more than the bandwidth
savings.

### 4. For Qwen 3.6 MoE on 36 GB M4 Max, the recommendation is:

| Scenario                           | Config                    | Why                                   |
|------------------------------------|---------------------------|----------------------------------------|
| Max decode speed, ≤128K context    | `-ctk f16 -ctv f16`       | No dequant overhead. KV fits fine.    |
| 128K+ context, speed matters       | `-ctk q8_0 -ctv q8_0`     | Near-f16 decode, 2× KV compression.   |
| 256K+ context, memory-critical     | `-ctk turbo3 -ctv turbo3` | Matches q8_0 decode w/ 5× V compress. |
| Everything else                    | `-ctk q8_0 -ctv q8_0`     | Generally best balance.               |

The session's Gemma 4 champion (`q8_0/iso3`) is **not recommended for this
MoE** — it's the slowest config tested here.

## Prefill notes

Prefill (pp512) is insensitive to KV type on this model — all 6 configs are
within 3% of each other (1135-1168 t/s). This is expected: MoE prefill is
bottlenecked by expert routing and the dense attention compute, not by KV
writes (which happen once per prompt token). The KV compression matters
during decode, not prefill.

## Compare to Gemma 4 (same fork, same hardware)

| Config               | Gemma 4 tg128 | Qwen 3.6 tg128 | Delta    |
|---------------------|--------------:|---------------:|:--------:|
| f16/f16             | 80.6          | 63.2           | −22%     |
| q8_0/q8_0           | 72.4          | 61.8           | −15%     |
| iso3/iso3           | 52.2          | 55.7           | +7%      |
| turbo3/turbo3       | 50.5          | 55.6           | +10%     |
| q8_0/iso3           | 59.5          | 58.3           | −2%      |

Interesting: iso3 and turbo3 actually decode **faster** on Qwen 3.6 than Gemma
4 in absolute terms, despite the model being 5× larger in total params. This
is because Qwen 3.6's smaller KV footprint means the dequant kernel runs on
proportionally less data per position, and the MoE A3B active-param structure
keeps per-token compute close to a 3-4B dense model.

**The cross-model picture**:
- f16 and q8_0 decode heavily favors dense models (Gemma 4 wins 22% and 15%)
- iso3/turbo3 decode is comparable or slightly faster on MoE (the rotation
  overhead is fixed per element, not per model param — MoE A3B absorbs it
  better)

If the goal is raw decode speed, **Gemma 4 E4B + f16 at 80.6 tg is the
fastest Metal configuration produced by this session's fork**.

## Raw log

Full bench log at `docs/qwen36-rotorquant-bench.log`.
