# Small Qwen and Gemma: FP16/BF16 training

Measured on 8 November 2026 on Apple M4, 16 GiB unified memory, macOS 15.5,
Temurin 20.0.2, PyTorch 2.8.0 and Transformers 4.51.3. Library code is master
`786668cb`, unchanged for this documentation follow-up. These are small training
configurations, separate from full-checkpoint inference measurements.

## Complete training steps

Lower milliseconds per step means faster training. Every cell is the median of
two fresh-process medians; each process retains 100 warm-up updates and 30 timed
updates. Engine order is DeepJ then PyTorch in the first pair and reversed in
the second. DeepJ recorded lower step times in both processes of every case.

| Measurement | Precision | DeepJ | PyTorch AMP |
|---|---|---:|---:|
| CPU training step (small Qwen) | FP16 (mixed) | **10.709 ms/step** | 81.029 ms/step |
| CPU training step (small Qwen) | BF16 (mixed) | **10.925 ms/step** | 81.784 ms/step |
| CPU training step (small Gemma) | FP16 (mixed) | **19.198 ms/step** | 60.181 ms/step |
| CPU training step (small Gemma) | BF16 (mixed) | **26.569 ms/step** | 60.583 ms/step |
| Metal / MPS training step (small Qwen) | FP16 (mixed) | **4.937 ms/step** | 18.008 ms/step |
| Metal / MPS training step (small Qwen) | BF16 (mixed) | **5.124 ms/step** | 17.484 ms/step |
| Metal / MPS training step (small Gemma) | FP16 (mixed) | **6.823 ms/step** | 19.228 ms/step |
| Metal / MPS training step (small Gemma) | BF16 (mixed) | **6.864 ms/step** | 16.324 ms/step |

These are mixed-precision training comparisons. The selected FP16/BF16 dtype
does not apply to every calculation or optimizer buffer.

## Work and precision policy

Both engines receive the exact initial checkpoint values and 64 aligned input
and target token IDs from the earlier small-model comparisons. Batch size is one;
loss is mean cross-entropy over all 64 targets. Qwen has 1,640,192 parameters,
two layers, width 256, feed-forward width 512, four query heads, two KV heads
and head dimension 128. Gemma has 1,247,488 parameters, two layers, width 256,
feed-forward width 512, four query heads, two KV heads and head dimension 64.
There is no dropout, checkpointing or KV cache. Configurations, IDs and initial
checkpoint hashes are in [inputs.json](inputs.json).

- DeepJ loads compact working parameters and activations through its existing
  model APIs. Gradients, AdamW master weights and first/second moments are FP32.
  Qwen selects `storageActivations=true`; Gemma uses its existing selected dtype.
- PyTorch keeps FP32 parameters and AdamW state and uses
  [PyTorch 2.8 autocast](https://docs.pytorch.org/docs/2.8/amp.html) around forward
  and loss, with backward outside autocast. Eligible operations use the selected
  dtype; normalization, embeddings and reductions follow PyTorch's AMP policy.
  Its initial FP32 master values are rounded through the selected dtype to match
  DeepJ's loaded initial values. Qwen uses SDPA; Gemma uses eager attention to
  retain its score softcap. AMP's operation-specific precision policy differs
  from DeepJ's compact working-parameter policy.
- FP16 uses the same fixed loss scale of 128 in both benchmark harnesses, with
  gradients unscaled before clipping. BF16 is unscaled. Every loss and gradient
  norm remains finite; no update is skipped. Fixed scaling is confined to this
  bounded benchmark harness.
- Both use AdamW: learning rate 0.0001, betas 0.9/0.999, epsilon 0.00000001 and
  weight decay 0.01. Parameter lists are sorted by name. Global gradient clipping
  uses maximum L2 norm one and conditional coefficient `1 / (norm + 1e-12)`.
  PyTorch uses its native norm and foreach-scaling operations and default AdamW
  `foreach`/`fused` choices. Complete steps include clearing gradients, forward,
  loss, backward, loss readback, unscaling when used, clipping, AdamW and final
  backend synchronization. Loading and state exports are outside timing.
- Both use ten CPU threads and `VECLIB_MAXIMUM_THREADS=10`; PyTorch uses one
  inter-op thread and MPS CPU fallback is disabled. Java uses `-Xmx1g -Xbatch
  -XX:CompileThresholdScaling=0.1`. Engines run sequentially; Metal/MPS checks
  run outside the sandbox through IntelliJ's terminal.

The FP32 rows on the summary page come from separate earlier runs with FP32
calculations and their documented training harness. These new rows compare the
two engines' mixed-precision implementations under the settings above.

## Numerical validation

Eight separate one-step audits compare initial logits, loss, raw unscaled
gradients before clipping and FP32 master weights after AdamW. The largest
initial loss difference is 0.0006089; the largest raw-gradient relative
L2 difference is 0.01652. Working weights are also captured after
each audit. Gradients and optimizer state remain FP32. Every measured run reduces
the objective substantially over 130 retained updates. Complete loss sequences,
errors and process timings are in [results.json](results.json).

AMP and compact working-parameter policies round at different points, so the
training trajectories are not expected to be bit-identical. The BF16 Gemma final
losses differ more than the FP16 losses; the raw sequences are retained. These
bounded checks do not establish identical long-run training or full-model
fine-tuning performance.

## CPU diagnosis

A separate three-step Qwen CPU operator profile explains the unusually slow
PyTorch compact rows on this Mac. `aten::mm` accounts for about 84% of self CPU
time in FP16 and BF16, compared with about 11% in FP32. Each profile records
135 matrix products; copy operations account for about 2% of compact self CPU
time. Matrix multiplication dominates this reference workload. The profiles do
not identify one specific native instruction or hardware cause, and their
instrumented timings are excluded from the table.

## Memory, evidence and reproduction

All 48 guarded processes complete: 16 numerical audits and 32 measured runs.
Each has a 60-second timeout, 2 GiB CPU RSS limit and 64 MiB swap-out guard.
No guard fires and no new system swap-out is recorded. RSS excludes some GPU
and shared-system memory. Background photo analysis and Spotlight workers are
resumed after timing; the user's existing training remains stopped.

The library revision used here already passed the full hardware build and CI in
[PR #200](https://github.com/KirstenAli/DeepJ/pull/200). This follow-up changes
documentation and measurement evidence only. Both benchmark sources compile/run,
all eight numerical audits pass, all 32 timed processes complete with finite
losses, and both independent process pairs favor DeepJ in each reported case.

[evidence.tar.gz](evidence.tar.gz) contains Java/Python harnesses, exact inputs,
configurations, raw result files, guard logs, audit checks and CPU profiles.
Use the earlier [Qwen fixture instructions](../qwen-training-2026-11-07/REPORT.md#reproduction-and-evidence)
and the included Gemma fixture generator, verifying their hashes against
`inputs.json`. Compile `CompactTrainingProbe.java` against the recorded DeepJ JAR.
The scripts accept model name, fixture directory, dtype, backend, output directory
and `audit`/`measure` phase. `measure.sh` records the exact invocation sequence;
adjust its local paths for another machine. Python is used for measurement only.
