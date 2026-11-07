# Full Qwen3-1.7B FP32 CPU prompt measurements

7 November 2026. Apple M4, 16 GiB unified memory, macOS 15.5;
Temurin 20.0.2, PyTorch 2.8.0 and Transformers 4.51.3.
DeepJ revision: `9792245592def3c8c1e1584e0ac36e6bafb9e0b0`.

## Completed comparison

This measures the **complete pretrained Qwen3-1.7B model**, with all
1,720,574,976 parameters loaded as FP32, batch size one and the same 64 prompt
token IDs. Both engines use a KV cache and read the complete 151,936-value
final-position logit row on each trial. There is no sampling, tokenization,
training or continuation decoding inside the prompt measurement.

Each fresh process performs eight warm-ups and five measured trials. Two
processes per engine run in the accepted order **DeepJ, PyTorch, PyTorch,
DeepJ**, reversing engine order between pairs. All five trials from each
accepted process contribute to its median; the README uses the median of the
two process medians. No individual timing sample is discarded.

| Pair | Order | DeepJ median (ms) | PyTorch median (ms) |
| --- | --- | ---: | ---: |
| 1 | DeepJ, PyTorch | 284.993 | 294.922 |
| 2 | PyTorch, DeepJ | 282.467 | 293.008 |
| **Median of process medians** | | **283.730** | **293.965** |

The rounded README row is **283.7 ms for DeepJ and 294.0 ms for PyTorch**.
DeepJ's prompt time is **3.5% lower** in this comparison and lower in both
pairs. The individual DeepJ trials range from 281.112 to 303.364 ms; PyTorch
trials range from 290.748 to 297.893 ms. These short runs describe this workload
and setup, rather than a universal or statistically established speedup.

## Timing and setup

Both engines load the same checkpoint revision,
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, used by the earlier full-model report.
Weights and model calculations are unchanged during this follow-up.
DeepJ uses its production CPU backend, ten executor threads and
`VECLIB_MAXIMUM_THREADS=10`. PyTorch uses ten CPU threads, one inter-op thread
and Transformers' SDPA attention implementation, without `torch.compile`.
Its loaded parameters are checked to be FP32 on CPU, with a tied output head.

DeepJ runs with `-Xmx9g -XX:+UseSerialGC -XX:NewRatio=8`. This collector and
heap configuration bounds the memory needed to load the full FP32 model on
the 16 GiB machine. It is a benchmark launch setting, not a library default or
an arithmetic change. Both launchers collect discarded loading allocations
once before warm-up. No explicit collection is requested between measured
trials. Loading, process startup, session setup, VM-counter collection and
printing are outside prompt timing; input preparation and full logit readback
remain inside the forward measurement.

The reference loader verifies byte-for-byte that the checkpoint's duplicated
`lm_head.weight` equals `model.embed_tokens.weight`, then loads their shared
FP32 storage once. This avoids temporarily allocating a second full output
matrix before tying it, and matches the tied architecture used by DeepJ.
All other tensors are restored normally; no weights are quantized, omitted
from the model or left on the meta device. Only the comparison loader changes;
no production source is modified for these measurements.

The earlier 292.8/286.5 ms comparison included 16 decode steps between prompt
trials and used an older DeepJ revision and different JVM settings. This is a
fresh paired prompt-only comparison. It replaces the README's FP32 CPU prompt
row without establishing an isolated before/after speedup from one particular
change. Cached decoding and the other README rows are not remeasured here.

## Memory checks and excluded attempts

IntelliJ and Chrome are closed before accepted runs. The stalled IDE and its
helper were using approximately 2.1 GB before they exited. Temporary Activity
Monitor windows are closed during timing. Training is not resumed, and no
image or video generation runs alongside the comparison.

Every measured trial records system `Swapins`, `Swapouts`, `Pageins` and
`Decompressions` immediately before and after its timed interval. The process
acceptance rule requires **all five measured trials to have unchanged swap-in
and swap-out counters**. All twenty accepted measured trials satisfy this rule;
all four accepted processes also have no new swap-out during loading or warm-up.
Other VM activity, including decompression and ordinary page-ins, remains in
the raw data; the measurements are not described as free of all VM activity.

A bounded launcher enforces a 100-second process timeout, a 512 MiB global
swap-out growth cap during loading/warm-up and a cancellation file. The Java
heap is explicitly bounded. Peak RSS is retained from child resource usage;
it is not used to claim a cross-engine memory advantage.

All excluded attempts are retained. A G1 loading attempt hit the loading swap
cap before the IDE was closed; another failed with an insufficient Java heap.
The original reference loader hit that cap while allocating the duplicated
output weights. The first completed tied-loader reference run had a 64 KiB
increase in the global swap-in counter during one measured interval. Its
entire five-trial group is excluded under the predefined rule, including its
otherwise usable samples. Exclusions depend on resource checks, not speed.

## Evidence and reproduction

[results.json](results.json) contains every accepted timing, warm-up, VM counter,
resource result, artifact/source hash and excluded attempt.
[evidence.tar.gz](evidence.tar.gz) contains the probes, exact input, reference
forward, loader, guards and raw logs. Model weights and application binaries
are excluded. The Python files are comparison tools, not DeepJ runtime code.

Build the recorded revision with `mvn package -Pcli`, compile
`FullCpuValues.java` against its CLI JAR, and substitute local checkout,
checkpoint and Python environment paths in `reverse_pair.py`. The first pair
uses the same commands with DeepJ first. Run engines sequentially and retain
all counter snapshots when deciding whether a process meets the acceptance
rule. Documentation validation checks the README row against the raw medians,
both model parameter counts, trial counts, run order, swap counters and links.
