# Rolling KV-cache qualification

25 October 2026, M4 MacBook Air, 16 GB unified memory, JDK 20, physical Metal.
Baseline: `47a4d09ccc3381d17eed83f9da3e2051f257fbfc`.
Qualified implementation: `94a4dbb3f3cf5076a86d24862d2dfa620e9e89e2`.
Subsequent evidence-only commits do not alter the tested Java, native or frontend source.

## Scope and result

Opt-in rolling generation for Orbit, Prism, Nova and Loop retains bounded KV storage and
continues RoPE positions after the window fills. Origin rejects rolling mode. Rebuild remains
the default. Training arithmetic, precision, saved weights, checkpoint format, sampling defaults
and the saved Instruction/Response prompt format are unchanged. No native kernels changed.

The local qualification is accepted after investigating the four flagged system-wide swap events.
Exact-SHA CI and PR review are separate merge requirements. No universal long-response quality
improvement or training speed improvement is claimed.

## Correctness and build

- Portable verification and the final physical full-coverage/CLI build passed. The latter ran
  1,234 tests with zero failures/errors; 11 separately opt-in timing tests were skipped.
- Java line/branch coverage: 92.27%/81.11%; Metal Java: 98.42%/92.27%; autograd: 100%/97.92%.
  Native: 96.77%/86.69%, with 100% functions/kernel selections. Frontend: 100% all metrics.
- CPU independent sliding-window references cover all four rolling model families, repeated
  Loop visits, multiple wraps, capacity one, repeated tokens, changed prompts and cleanup.
  Twenty new physical Metal numerical cases passed. Actual WebView bridge and alternatives
  preserve the selected mode. Defaults and unsupported-model errors are tested.
- Full-size Prism 90M, Nova 100M and Loop 100M pre-eviction logits matched master bit for bit:
  11 positions and 180,224 values per model, in both candidate modes.
- Their losses, complete gradient hashes and updated-weight hashes matched exactly after three
  updates, including after rolled inference. Two hundred real TinyStories updates and five
  validations per model also matched exactly between master and candidate.
- The independent noncircular post-eviction oracle checked 65 positions/1,064,960 values per
  model. It gathers the latest KV rows and uses a larger fixed RoPE table, not circular storage
  or on-demand position rows. The original per-value bound, `2e-5 + 2e-4 * abs(reference)`, was
  unchanged. Worst fractions of that bound were 0.121615 Prism, 0.035351 Nova and 0.050719 Loop.
- Sixteen real Frisky short samples (eight prompts, two seeds) matched exactly in all three
  master/default/rolling-before-eviction cases. Frozen source and binary manifests verified;
  working `src/` and `native/` match the frozen candidate exactly.

## Timings and memory

Fresh JVMs, balanced order, two warm-up and seven measured requests per case, three rounds.
The table reports the median of paired candidate/baseline ratios, not a pooled best run.
Each request includes prefill, decode, readback, cache allocation and cleanup. Post-window cases
use a 1,024-token prompt plus 16 decode steps. These compare different post-eviction semantics.

| Model | Default pre-window time change | Rolling pre-window time change | Rolling post-window time reduction |
|---|---:|---:|---:|
| Prism 90M | +3.82% | -0.69% | 87.91% |
| Nova 100M | +0.32% | +2.18% | 88.18% |
| Loop 100M | -0.89% | -0.71% | 87.23% |

Every paired-median pre-window change stayed within the predefined 5% review threshold.
Prism had an individual default round +10.02%; all raw pairs remain in `analysis.json`.
Three small CPU training pairs gave ratios 0.98802, 0.99181 and 0.98784 (median about -1.20%).
These are bounded local measurements, not claims about every machine or workload.

Requested native peak was unchanged in every matched timing pair. Median process-footprint
ratios were within the predefined 10% review threshold. Java allocation, native requested
storage and process footprint are distinct measurements; none is advertised as total GPU memory.

Endurance covered Prism/Nova 2,048 tokens plus 50 closed requests each and Loop 4,096 tokens plus
200 requests. Native retained storage stayed constant; after close it was 363,137,024 bytes
Prism, 405,137,664 Nova and 407,794,176 Loop. There was no retained-cache growth or global
swapout increase in those endurance runs.

## Swap findings, investigation and remaining shape limitation

The original analysis deliberately exits with review findings for global OS swapout changes:
Prism candidate state 14,196 pages; Loop baseline pre-window 30,992; candidate rebuild 20,144;
candidate rolling 5,256. These counters are system-wide, not attribution to the Java process.
They are preserved, not erased by successful repeats.

Fresh controls reproduced Loop events in both old and new code: candidate rolling 23,876 pages,
baseline 5,556, candidate rebuild zero. Both Prism state repeats had zero and exactly identical
losses/gradient/weight hashes. The full-size inference results remained exact.

A separate per-request shape control identified an existing memory-heavy path: the Metal Flash
kernel requires complete 32-row tiles. A 1,023-token prefill therefore uses the quadratic
fallback; 1,024 tokens uses Flash. Both revisions produced exactly the same requested-native peaks:

| Prefill | Flash selected | Native peak bytes | Native retained after close |
|---|---|---:|---:|
| 1,024 | yes | 2,814,360,840 | 407,794,176 |
| 1,023 | no | 5,522,017,596 | 407,794,176 |

Each length was repeated without retained growth; final backend release left zero buffers.
The candidate/baseline process peaks were 6,893,095,960/6,882,724,696 bytes. Neither of these
instrumented controls increased global swapouts. The higher fallback peak is an existing
supported-shape limitation, not a new rolling-cache allocation regression. It is not claimed
fixed here, and the OS data does not prove exactly which process or allocation triggered paging.
No timing case was affected by a recorded global swapout event.

## Checkpoint-specific quality

The read-only actual Frisky checkpoint was tested on four held-out 1,088-token sequences,
260 next-token targets covering the first eviction. Candidate default losses matched master
exactly, including every first pre-eviction token. Mean NLL was 1.49385880140 rebuild versus
1.50792391209 rolling, a +0.01406511069 difference, below the predefined +0.1 review threshold.
Per-sequence differences were +0.04446209, +0.00562054, -0.00374110 and +0.00991891.
This small probe is not a broad perplexity benchmark or a factual-accuracy evaluation.

The final free-generation check used two held-out passages, two seeds and both modes, with
the saved prompt template, unchanged temperature 0.8/top-k 40/penalty 1.0 and normal EOS stops.
Prompts had 1,024/1,023 tokens, and every sample crossed the boundary. Outputs contained 59–160
new tokens; early EOS was preserved. Manual review found understandable story-style prose but
repetition and unsupported narrative changes in both modes. Rolling was not consistently better;
one rolling sample was conspicuously repetitive. It remains opt-in, with no long-context quality
guarantee. Private passage/output text is excluded from the public evidence.

## Harness failure and evidence provenance

The first runner completed 79 cases, then the sample harness failed before generation because
it passed an SFT directory to a loader expecting `model.dj`; the actual file is `model-final.dj`.
The failed invocation/log remain preserved. Recovery verified frozen manifests and reran only
the remaining 11 cases with the explicit checkpoint filename. No model was renamed, copied or
written; no production code, test tolerance or frozen harness was changed.

The analysis input consists of those 79 original successful cases plus 11 recovery cases, with
per-file provenance. The original four review findings remain in `analysis.json`; their bounded
investigation is recorded in `review-resolution.json`. Additional controls and free-generation
checks are separate from the 90-case suite. `evidence.tar.gz` contains sanitized scripts,
harness sources and metric logs; model files, token dumps, dataset passages, private generations
and credentials are excluded.
