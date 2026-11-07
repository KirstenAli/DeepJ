# CPU causal softmax: retain completed native rows

7 November 2026. Apple M4, 16 GiB unified memory, macOS 15.5,
Temurin 20.0.2. Baseline: `2b602a24`.

## What changed

The native FLOAT32 causal softmax uses vector exponentials and checks their
distance from FLOAT32 rounding boundaries. Previously, one sensitive value
caused the caller to recompute the entire matrix in Java. In the five profiled
prompts, every matrix triggered that whole-matrix fallback.

The native kernel now reports which rows need scalar evaluation. The CPU backend
reuses its existing Java row implementation only for those rows and retains
the other native results. The exponential guard, masking sentinel, reduction
order and Java scalar arithmetic are unchanged. This also preserves existing
NaN, infinity and extreme-input behavior. The original four-argument native
helper keeps its previous boolean acceptance contract.

The implementation reuses the existing native row dispatcher and Java scalar
row function. It adds no model-specific route, duplicate scalar kernel,
architecture change, optimizer change, Metal change or build-gate exemption.
The compact FP16/BF16 softmax implementation is unchanged.

## Repeated component measurements

The component fixture contains the first pretrained Qwen3-1.7B block, with its
full dimensions: width 2048, MLP width 6144, 16 query heads, eight KV heads and
head width 128. It retains only the 25 embedding rows needed by the fixed input.
Weights come from revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` and are loaded
as FLOAT32. The prompt has 64 token IDs and batch size one.

Each fresh process performs 60 warm-ups and 25 timed trials. Each trial processes
the prompt and then the same 16 continuation token IDs with a KV cache, reading
the complete fixture logit row after every forward. There is no sampling.
Four before/after pairs alternate execution order. Both versions use ten CPU
threads, `VECLIB_MAXIMUM_THREADS=10`, a 1 GiB heap and the same probe.

| Pair | Order | Before prompt (ms) | After prompt (ms) |
| --- | --- | ---: | ---: |
| 1 | Before, after | 10.711 | 9.729 |
| 2 | After, before | 10.627 | 9.827 |
| 3 | Before, after | 10.593 | 9.850 |
| 4 | After, before | 10.468 | 9.915 |
| **Median of process medians** | | **10.610** | **9.839** |

Prompt time is **7.3% lower**, with an improvement in all four pairs. This is
a single-block component result, not full-model throughput or a new PyTorch
comparison. Separate diagnostic profiles measured the causal softmax operation
at 0.715 ms before and 0.131 ms after; instrumentation timings are excluded
from the paired table. All captured prompt and decode logits in those profiles
are identical between versions. The decode operation route is unchanged;
no new cached-decoding speedup is claimed.

## Correctness and training controls

Tests compare outputs exactly against the independent scalar implementation,
including random inputs, masking, extreme values, NaNs, infinities, serial and
parallel execution. New tests verify that only sensitive rows request fallback,
that causal positions remain correct across multiple head groups, and that
the original helper's acceptance contract is preserved.

Separate small Qwen, Loop and Prism training controls retain identical loss
sequences and final parameter/gradient SHA-256 hashes. These controls use width
64, vocabulary 128, one 32-token sequence, one executor worker, 20 warm-up
updates and 11 measured updates. They include backward, clipping and AdamW.
They establish correctness for those controls rather than a training speedup.

`mvn -o verify -Pcli -Dgemma.test.metal=true` passes through the authorized
IntelliJ terminal on physical Metal: **3,014 tests, zero failures/errors,
57 optional skips**. Checkstyle, PMD, duplication, frontend and native quality
gates pass. Core line/branch coverage is 96.21%/85.89%; autograd is
99.65%/96.09%. An additional Java-only softmax check passes: 21 tests,
zero failures/errors, 11 native-only skips.

## Full-model follow-up

The full FP32 checkpoint was attempted with the same 64-token prompt, eight
warm-ups and five planned measured trials, two processes per engine and
reversed before/reference/after order between rounds. Java explicitly collects
discarded loading buffers once before warm-up; the reference loader likewise
collects loading temporaries. No collection is requested between timed trials.

The unchanged baseline triggers the 64 MiB system swap-out growth guard before
measured trials. Reducing the G1 region size was investigated: the default
10 GiB heap uses 8 MiB regions, and the weight-array sizes account for about
1.65 GB of allocation padding. With 1 MiB regions the corresponding calculated
padding is about 207 MB. The smaller-region attempt with an 8 GiB heap fails
during loading; a 10 GiB heap still triggers the swap guard during loading.
All these processes are terminated by the launcher or exit on the loading error.
The resource guards are not relaxed to obtain a timing.

There is no accepted new full-model comparison. The README retains the last
completed FP32 CPU prompt figures, 292.8 ms versus PyTorch's 286.5 ms; this
component improvement does not establish that the full-model gap is closed.
All paused background jobs are resumed. Training remains stopped, and no
image or video generation is part of this work.

## Reproduction

[results.json](results.json) contains every paired component sample, training
control, artifact/source hash, validation result and excluded full-model attempt.
[evidence.tar.gz](evidence.tar.gz) contains the probes, fixture extractor,
configuration, fixed inputs, launch scripts, guards, raw logs and build reports.
The baseline softmax audit uses the original internal JNI signature and must
be compiled against the baseline build. Build each recorded version with
`mvn package -Pcli`, compile the probes against its CLI JAR, substitute local
paths and run `balanced.py` to reproduce the component comparison. Model weights
and application binaries are excluded from the archive. Python files are
comparison and fixture tools, not DeepJ runtime code.
