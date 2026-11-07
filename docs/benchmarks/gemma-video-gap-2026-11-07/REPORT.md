# Gemma training and media performance checks

7 November 2026. Apple M4, 16 GiB unified memory, macOS 15.5 and Temurin 20.0.2.
These checks follow `797ff72c` on `feature/shared-tensor-performance`.

## Small Gemma training comparison

A **1,247,488-parameter Gemma configuration** uses two layers, width 256,
MLP width 512, four query heads, two KV heads, head width 64 and vocabulary 256.
Both engines start from the same generated checkpoint and train on the same
fixed next-token batch: one sequence with 64 input tokens and 64 targets.

All parameters and calculations in this training comparison use **FP32**.
A complete timed step includes zeroing gradients, forward, mean cross-entropy,
backward, global gradient clipping at norm 1, AdamW and synchronization.
AdamW uses learning rate 0.0001, betas 0.9/0.999, epsilon 0.00000001 and weight
decay 0.01. PyTorch uses built-in `clip_grad_norm_` with
`error_if_nonfinite=True`; both engines check finite losses and gradients before
updating. Its clipping epsilon is 0.000001; DeepJ's is 0.000000000001.
PyTorch's `foreach` and `fused` options retain their defaults.
Both engines use ten CPU threads; PyTorch uses one inter-op thread.

Each engine/backend runs in **two fresh processes**, with 100 warm-up steps
and 30 timed steps per process. The first pair runs DeepJ before PyTorch;
the second reverses the order. Each cell below is the median of all 60 timed
steps. Lower milliseconds per step means faster training.

| Backend | DeepJ (ms/step) | PyTorch (ms/step) |
| --- | ---: | ---: |
| CPU | 9.65 | **7.50** |
| Metal / MPS | **10.55** | 12.26 |

DeepJ's pooled Metal median is **14.0% lower**;
its CPU median is **28.7% higher**.
The reference is PyTorch 2.8.0 with Transformers 4.51.3.
These pooled medians do not establish a consistent Metal winner: the first
fresh-process pair measured 10.33 ms for DeepJ and 10.27 ms for PyTorch; the
second measured 10.55 and 16.92 ms. PyTorch's Metal timings varied substantially.
These short runs establish results for this small configuration. GPU timing
varies between fresh processes; every step and process median is retained in
[results.json](results.json).

Both engines start at loss 5.5921097 and finish near 0.2493 after 130 steps.
The maximum loss-sequence difference is 0.000381 on CPU and 0.000040 on Metal.
The independent Transformers embedding-gradient oracle also passes. Floating-point
reduction and optimizer rounding differ between engines, so the trained states
are not claimed to be bit-identical.

## Shared implementation changes

- Dense FLOAT32 gradient norms now use the existing parallel Metal sum-of-squares
  kernel regardless of tensor rank. Previously, a one-dimensional parameter
  sent the entire gradient list through serial per-tensor layout reductions.
  The small Gemma Metal step fell from about 112 ms before the changes to the
  timings above. Matrix-only inputs retain their existing reduction kernel;
  vector/matrix/higher-rank views produce identical norm bits in the new test.
- Strided CPU pointwise operations advance physical addresses along each row,
  avoiding per-element coordinate division. Dense FLOAT32 last-axis softmax and
  its backward pass reuse the existing matrix kernels. Contiguous indexed
  gradient updates add whole trailing blocks in the original repeated-index order.
- Gemma training uses DeepJ's existing differentiable RMSNorm and matrix
  projection paths. The model architecture, loss and AdamW settings are retained.
- Gemma reuses the per-forward rotary-table cache for both compact formats.
  Its BF16 rounding convention has a distinct cache key. Cached and uncached
  rotary outputs and gradients match exactly across layer-scope cleanup.
- Checkpoint payloads are read in endian-aware bulk chunks. New shared model
  weights upload their private staging directly to the selected backend, avoiding
  a second same-dtype device weight allocation and assignment. Dtype-conversion
  scratch is retired through the existing buffer-lifetime mechanism.
- Compact affine RMS inference combines gain conversion and optional unit offset
  in the existing normalization kernel. The optimization is restricted to an
  explicit bounded inference scope, preserving the staged image encoder's
  established arithmetic.
- The shared compact cached-attention interface now supports a score softcap and
  local-window starting position. Gemma chat uses this inference-only operation;
  the default cached-attention specialization retains its existing arithmetic.
  Training continues through autograd.

## Full Gemma inference checks

The complete **2,614,341,888-parameter** checkpoint supplied with SANA-Sprint is
used for inference. Both engines receive the same 21-token formatted prompt and
16 fixed continuation token IDs, with batch size one, KV cache capacity 128,
full 256,000-logit readback at every step and no sampling. Loading and tokenization
are outside the timed sections. There are three warm-up trials and three timed
trials per process, with two fresh processes per engine/dtype and reversed
engine order in the second pair. PyTorch uses eager attention to preserve the
score softcap; MPS CPU fallback is disabled.

These paired results include checkpoint and rotary-cache improvements, before
the subsequent capped-attention fusion:

| Precision | DeepJ decode (tokens/s) | PyTorch decode (tokens/s) | DeepJ prompt (ms) | PyTorch prompt (ms) |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 15.35 | 15.59 | 98.7 | 95.4 |
| BF16 | 15.12 | 15.47 | 105.1 | 95.5 |

A subsequent single-process fusion pilot measured 15.32 tokens/s for FP16 and
15.33 for BF16. It does not establish a decode-throughput win over PyTorch.
The remaining prompt and decode gaps are retained in the results rather than
relabelled as faster performance.

All 17 vocabulary rows were captured: 4,352,000 logits per dtype. The capped
cached-attention kernel produces **byte-identical captured logits** to DeepJ's
unfused implementation for these inputs. Relative to PyTorch:

| Precision | Mean absolute error | Maximum absolute error | Matching argmax rows |
| --- | ---: | ---: | ---: |
| FP16 | 0.0056843 | 0.046875 | 17/17 |
| BF16 | 0.0335128 | 0.375 | 16/17 |

The BF16 difference is the previously recorded tie: DeepJ gives tokens 2185 and
6011 scores of 18.375; PyTorch gives them 18.25 and 18.375. Fixed continuation
inputs keep subsequent rows comparable.

## Original fox restored

The FP32 SANA-Sprint run uses the original prompt and settings:

> A red fox in a snowy forest, soft morning light, detailed wildlife photograph

**1024 × 1024, two steps, guidance 4.5, seed 42.** Restoring the staged image
normalization policy reproduces the original README fox PNG byte-for-byte.
Generation took 76.63 seconds, including stage loading. The visibly different
previous image was the BF16 output.

SHA-256: `0231f74c33d19bd28520ccbafac61ce302a7ed20e3353c41377800105c89a6c2`.

![Original FP32 fox](fox-original.png)

## Tiny video comparison

Only **192 × 128, five frames and one denoising step** were used. The full
pretrained checkpoints receive identical prompt IDs and Java-generated starting
noise, guidance 6 and seed 42. Gemma and the Transformer use BF16; sampler state
and the Wan decoder use FP32 in both engines. The reference uses PyTorch 2.8.0,
Diffusers 0.42.1 and Transformers 5.22.1. MPS CPU fallback is disabled.

The first successful matched Python pipeline took **16.06 seconds** including
stage loading. DeepJ took **28.04 seconds** after bulk reading; an explicit
larger weight-cache probe took **25.33 seconds**. PyTorch batches positive and
negative guidance together; DeepJ evaluates them sequentially. These are single
cold pipeline runs. No long or full-size video was generated.

An isolated Wan check used the same raw FP32 noise tensor in both engines,
with five output frames at 192 × 128. The first call includes lazy decoder
weight loading and initialization; the second reuses the loaded weights.
PNG writing is outside the timing.

| Decoder call | DeepJ (s) | PyTorch (s) |
| --- | ---: | ---: |
| First | 8.49 | **2.46** |
| Second | 7.97 | **1.88** |

The decoder remains a substantial performance gap. An isolated bounded
frame-batching prototype measured 8.44 and 7.87 seconds and was not promoted.
Production video decoding and its memory-bounded frame schedule are unchanged
by that experiment.

## Validation and evidence

The final `mvn -o verify -Pcli -Dgemma.test.metal=true` build passes **2,980 tests**
with zero failures/errors and 57 optional skips. Checkstyle, PMD, duplication,
frontend, native code quality and Java coverage gates pass. Core Java line/branch
coverage is 96.19%/85.89%; autograd coverage is 99.65%/96.09%. The rebuilt CLI
starts successfully. Focused Metal checks include softcap/window results, all
representable compact input values, malformed native commands and independent
model/gradient references. The final artifact hash is in `results.json`.

All accepted model comparisons recorded no new swap-out. An excluded prefix
submission prototype was stopped by the memory guard before it yielded usable
timings. A 2 GiB heap attempt could not allocate Gemma's largest checkpoint
array; accepted full-model comparisons use 3 GiB. Other prepacked-loading and
MPSGraph optimization-level experiments did not improve the comparison and
were not promoted.

The earlier manual-clipping reference is retained separately in `results.json`;
the training table uses the built-in PyTorch clipping API.

Photo analysis and Spotlight were temporarily paused with permission and resumed
after each batch. The user's training remained stopped. RSS and swap guards bound
standalone model processes; RSS alone is not total GPU/system memory.

[results.json](results.json) contains every accepted training step, process
median, inference trial, pipeline stage, artifact hash and numerical summary.
`evidence.tar.gz` contains probes, independent Python references, fixture
configuration/input, source changes and validation logs. Python is used only
by the comparison tools, outside DeepJ's Java runtime.
