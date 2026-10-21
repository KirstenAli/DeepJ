# Tests and coverage

Run the portable build with JDK 20:

```sh
mvn clean verify
```

The build runs Java tests, frontend behavior tests, formatting, lint, short-function checks and duplication checks. It fails if coverage falls below the configured minimum. Coverage is collected only during tests, not during training or normal library use.

## Rules also apply to tests

Java production and test sources, including test-support classes and performance-test helpers, share the Checkstyle and PMD rules. These include method length, statement count, complexity, nesting and duplication. Tests have no blanket rule exemptions.

Frontend JavaScript tests and helpers are discovered recursively under `src/test` for formatting, ESLint and duplication checks. HTML/CSS test files receive the same applicable formatting, markup and stylesheet checks as production assets. Frontend test execution also includes nested `.test.mjs` files.

Native source and test directories are scanned recursively for `.h`, `.hpp`, `.c`, `.cc`, `.cpp`, `.m`, `.mm` and `.metal` files. Portable tests enforce the 20-nonblank-line function limit, 120-character line limit, loop braces, one statement per line and duplication rules, including copies across files. On macOS, compiler-based checks additionally enforce function length, cognitive complexity and nesting across all discovered native sources and headers. Declaration-only headers are parsed too, even when they define no executable functions.

Deliberately invalid source samples used to test a checker are stored as string data, not exempted implementation files. Coverage percentages measure production code exercised by the tests, not the test code itself.

## Control-flow checks

`mvn clean verify` also rejects methods with cognitive complexity above 15. PMD checks production and test Java, including the Java Metal backend; ESLint checks frontend JavaScript, its tests and build scripts.

The Java nesting rule allows up to three mixed control-flow levels, counting `if`, loops, `switch` and `try` together. Else-if chains stay flat, and nested methods and lambdas start a new nesting count. Existing stricter Java `if`/`for` checks remain enabled. JavaScript keeps its existing two-level nesting limit.

The rules live in `config/pmd.xml`, `config/checkstyle.xml` and `eslint.config.mjs`. Regression tests check both acceptable examples and deliberately complex examples, including the 15/16 cognitive-complexity boundary. These are build-time checks, not runtime instrumentation.

### Native host and Metal shaders

On macOS, `mvn clean verify` automatically activates `native-code-quality`. It enforces a 20-nonblank-line function limit, cognitive-complexity maximum of 15 and mixed control-flow nesting maximum of 3 on native host functions, extracted shader functions and native test helpers. No GPU work is executed by these static checks. Xcode Command Line Tools provide the Clang parser and macOS headers; use a macOS 15 or newer SDK for DeepJ's existing Metal compile options. Missing tools, parse errors, missing kernel definitions or rule violations fail the build.

Clang parses generated analysis-only copies under `target/quality/native`. Shader type declarations and address-space qualifiers are supplied by `src/test/native/metal-analysis-shim.h`, while the original function bodies are preserved. This checks C++-compatible control flow, not all Metal language or GPU numerical semantics. The production source, bundled library, training process and checkpoints are untouched.

The native metric uses Clang's syntax tree, not brace or keyword counting. It accounts for nested branches and loops, switches, catches, ternaries, logical-operator sequences and direct recursion. Else-if chains stay flat; unbraced nesting still counts. Like common compiler-based checks, inactive preprocessor branches and cross-function recursion cycles are not measured. Limits live in `config/native-quality.json`; per-function results and original source lines are written to `target/quality/native/report.json`.

Native regression tests verify the scoring and boundary conditions against real parsed C++ fixtures. They also test shader extraction and failure handling. Non-macOS builds retain the Java/frontend checks; a separate macOS CI job checks native host and shader control flow on every pull request without requiring a GPU.

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
