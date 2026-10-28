# Coding standards and regression testing

DeepJ changes must be readable, tested and checked against the previous implementation.
A green unit-test build alone does not establish numerical correctness, performance or memory safety.
Unexplained differences must be reproduced, investigated, corrected and retested before approval.

## Enforced coding standards

Production and test code are both checked. Prefer focused functions, explicit ownership,
clear names and early returns. Do not weaken a rule, exclude a failing file or duplicate an
implementation to make a change pass.

| Scope | Enforced limits |
|---|---|
| Java | Methods and constructors: 20 non-empty lines; 15 executable statements; cyclomatic complexity 8; NPath complexity 80; cognitive complexity 15. |
| Java nesting | Nested `if` and loop depth: 2 each. Mixed control-flow nesting: 3. |
| Java layout | 120-character lines, with URL exceptions; 400-line files; class NCSS 200; one statement per line; braces around loops. |
| JavaScript | Functions: 20 non-blank lines; 15 statements; cyclomatic complexity 8; cognitive complexity 15; nesting depth 2. |
| Native host code and Metal shaders | Functions: 20 non-empty lines; cognitive complexity 15; mixed control-flow nesting 3. |

The definitions are in [Checkstyle](../config/checkstyle.xml), [PMD](../config/pmd.xml),
[ESLint](../eslint.config.mjs) and [native quality configuration](../config/native-quality.json).
Native analysis covers source and helper headers under `native/` and `src/test/native/`,
including shader strings and compact shader variants. It runs on macOS with the SDK headers.

Prettier checks the configured frontend, test and quality-tool files. ESLint, Stylelint and
HTML Validate check JavaScript, CSS and HTML. Java copy/paste detection rejects matches from
100 tokens; the frontend checker rejects matches from 100 tokens and 5 lines. These checks
are language-specific, not a claim that every language uses an identical analyser.

## Build and coverage gates

With JDK 20 or newer selected, run:

```bash
mvn clean verify
```

This runs portable Java tests, Java quality and coverage gates, and frontend checks and tests.
On macOS it also runs native source-quality checks. Maven installs its own local Node runtime.
Hardware-dependent Metal and desktop tests are opt-in; skipped tests are not evidence that
those paths passed. The portable build remains usable without an Apple GPU or desktop session.

On a supported Apple Silicon Mac with Metal and an available desktop session, also run:

```bash
mvn clean verify -Pfull-coverage
```

This requires actual Metal, enables desktop tests and builds a separate instrumented native
test library. It adds full Java, Metal integration and native coverage gates; it does not
replace the larger, matched regression and performance qualification described below.
Rebuild the bundled production library with `./native/build-macos.sh` after JNI or shader edits.

Minimum coverage is enforced by [Java/native configuration](../config/coverage.json) and
[frontend configuration](../config/ui-coverage.json):

| Coverage scope | Minimum |
|---|---|
| Portable Java | 85% lines, 70% branches |
| Core Java | 90% lines, 75% branches |
| Full Java and Metal Java integration, each | 90% lines, 75% branches |
| Autograd Java | 95% lines, 90% branches |
| Native host/JNI | 85% lines, 70% branches, 90% functions |
| Metal shader pipeline selections | 100% |
| Frontend JavaScript, per file | 95% lines, statements, branches and functions |

Portable/core coverage has explicit package exclusions for hardware or desktop paths; the
full profile checks them separately. Native line coverage measures executed host/JNI code.
Shader selection coverage confirms that each tracked pipeline was selected, not that every
GPU instruction, shader branch or possible input shape was exercised. Numerical shader tests
are required as well. Test sources are quality-checked; coverage percentages measure production code.

## Strict numerical and state regression

For tensor, autograd, model, optimizer, cache or backend changes:

- Compare against independent reference calculations and finite-difference gradients, not
  only another path sharing the changed implementation.
- Test shapes, broadcasting, empty and singleton axes, strided/offset views, alias lifetimes,
  mutation checks and invalid inputs. Compare CPU and actual Metal forward and backward results.
- Exercise all affected model families, shared parameters and repeated Loop visits. Compare
  dense batches with independent per-sequence calculations, including response-only masks.
- Freeze the old and new source, binaries, inputs and seeds. Check losses, logits, gradients,
  updated weights and optimizer state. Use baseline self-controls to test repeatability.
- Check real checkpoint files: model settings, prompt format, tokenizer references, tied
  parameters, dataset cursor, step counters, moments and any master weights. Test actual file
  restore and continued training, not just an in-memory copy. Compare complete saved bytes
  when the file format is expected to remain identical.
- Test optional dtypes separately against correctly rounded references. Verify real storage
  widths, promotion, gradients, optimizer precision and exact typed save/resume. Preserve
  existing default-FLOAT32 bounds; never loosen them to accept compact-precision differences.

For core migrations, qualification includes full-size Prism 90M, Nova 100M and Loop 100M
comparisons at a 1,024-token context, alongside smaller operator and all-family tests.
Record every discrepancy, including one below an existing tolerance, and investigate its cause.
Bitwise identity is checked where the unchanged computation is expected to be reproducible;
any justified floating-point difference needs explicit evidence, not silently relaxed bounds.

## Performance, memory and endurance

CPU and Metal are separate acceptance targets. Match model, data, seed, optimizer, context,
batch size, JVM/native options and cleanup policy. State any smaller-context CPU workload
explicitly; do not advertise it as a full-context result.

- Warm up before measuring. Repeat independent processes in balanced old/new order and retain
  individual timing distributions. Use old-versus-old and candidate self-controls when a gap
  needs investigation. Do not treat correlated steps as independent experimental repeats.
- Keep startup, validation, measured steps and instrumented diagnostics separate. Profiles
  help locate a cost; they are not substitutes for uninstrumented throughput measurements.
- Investigate consistent slowdowns at operation and allocation level, fix their causes and
  rerun the matched checks. "Nearly the same speed", an inconclusive statistical result or
  faster rounds elsewhere does not excuse an unexplained regression.
- Measure allocation, heap/GC, native live and peak buffers, process RSS and system memory
  pressure. Distinguish a process's memory from system-wide swap activity; do not assign an
  OS swap event to one process without evidence.
- Run real-data endurance with held-out validation and finite, positive gradient checks.
  Compare equivalent cleanup boundaries for retained growth. Core qualification uses
  200-update FLOAT32 runs, plus separately defined dtype stability and save/resume checks.
- Use explicit per-run memory, disk and time guards. Stop unsafe tests, preserve the failure
  and investigate. Do not lower a guard, shorten the original workload or add production
  forced GC to manufacture a pass. Never run benchmarks alongside real training.

Performance claims must be scoped to the tested workload and hardware, with uncertainty
reported honestly. No finite benchmark proves identical timing under every machine condition.

## Evidence and approval

Keep source/binary hashes, commands, raw logs, numerical results, timing distributions,
memory telemetry and failed attempts. Investigate test-helper failures too; rerun corrected
helpers without relabelling earlier failures as passes. Do not edit frozen benchmark copies.

Before a PR, resolve identified gaps, complete the relevant strict qualification, update
documentation and verify CI on the exact proposed commit. Record unavailable checks explicitly;
do not present an incomplete required check as passed. Do not merge or release without the
separate authorization required for that action.
