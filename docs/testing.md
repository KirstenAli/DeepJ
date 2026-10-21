# Tests and coverage

Run the portable build with JDK 20:

```sh
mvn clean verify
```

The build runs Java tests, frontend behavior tests, formatting, lint, short-function checks and duplication checks. It fails if coverage falls below the configured minimum. Coverage is collected only during tests, not during training or normal library use.

## Coverage gates

| Scope | Lines | Branches | Other minimums |
|---|---:|---:|---|
| Portable Java, including examples | 85% | 70% | Hardware-only packages checked separately |
| Core Java, excluding examples | 90% | 75% | Same hardware split |
| All Java in the full hardware build | 90% | 75% | No production classes excluded |
| Java Metal backend in the full hardware build | 90% | 75% | Separate package gate |
| Frontend JavaScript, per file | 95% | 95% | 95% statements and functions |
| Native Objective-C++ host code | 85% | 70% | 90% functions |
| Metal shader pipeline selections | — | — | 100% of declared kernel entry points |

Java gates are defined in `config/coverage.json`; frontend gates are defined in `config/ui-coverage.json`. Counter totals are aggregated by executed/total lines and branches, not by averaging package percentages.

The portable Java gate excludes `io.github.kirstenali.deepj.tensor.metal` and `io.github.kirstenali.deepj.chatui`, because those packages require a GPU or desktop session for their complete test suite. They remain visible in the full JaCoCo report; their unexecuted code is not presented as covered. The core gate additionally excludes `examples`. The full hardware gate includes every production Java package, including examples and desktop code.

HTML/CSS are checked for validity, style and duplication. Frontend JavaScript runs against the real chat HTML in a DOM test environment. Those tests do not verify rendered pixel layout; the separate JavaFX/WebView tests exercise the actual desktop browser bridge and layout.

## Full Metal and desktop coverage

On Apple Silicon macOS with a working Metal device, a desktop session, Xcode Command Line Tools and JDK 20:

```sh
mvn clean verify -Pfull-coverage
```

Run this when training is paused so the GPU work does not compete. The profile explicitly requires Metal and enables the desktop tests. Missing GPU access is a failure, not a successful coverage result based on skipped tests.

The profile compiles a separate instrumented native library from a generated copy of the source. Only the full-coverage test classpath uses it. The normal bundled native library, production source and training checkpoints are not modified. Full-coverage test classes use a separate output directory, so the instrumented library cannot accidentally remain on the portable test classpath.

Clang/LLVM reports native host line, branch and function coverage. GPU shader bodies are compiled from a runtime source string and are **not** measured by Clang's host coverage. A separate test-only counter checks which shader pipelines were selected during functional tests. This is kernel-entry selection coverage, not shader line or branch coverage, and does not substitute for CPU/Metal numerical assertions. Shader source changes remain subject to numerical and native-source tests.

Fresh Java execution data and native profiles are required for each run. The build must not pass using stale coverage from an earlier run. Manual performance benchmarks are not part of the coverage requirement.

## Reports and automation

| Report | Location |
|---|---|
| All production Java | `target/site/jacoco/index.html` |
| Checked Java scopes | `target/coverage/java-summary.json` |
| Frontend JavaScript | `target/coverage/frontend/index.html` |
| Frontend LCOV | `target/coverage/frontend/lcov.info` |
| Native host coverage | `target/coverage/native/html/index.html` |
| Native machine-readable report | `target/coverage/native/coverage.json` |
| Metal kernel selections | `target/coverage/native/kernel-summary.json` |

GitHub Actions runs the portable checks on pull requests and pushes to `master`, and uploads the reports. A full native/GPU result requires running the hardware command on a suitable Mac; a hosted Linux CI result does not claim native or desktop coverage. Native gate values are policy thresholds until confirmed against an actual full hardware run.
