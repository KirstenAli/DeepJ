<p align="center">
  <img src="./deepj_logo.svg" alt="DeepJ logo" width="170">
</p>

# DeepJ

<p align="center">
  <a href="https://central.sonatype.com/artifact/io.github.kirstenali/deepj"><img src="https://img.shields.io/maven-central/v/io.github.kirstenali/deepj.svg?label=Maven%20Central" alt="Maven Central"></a>
  <a href="https://adoptium.net/"><img src="https://img.shields.io/badge/Java-20%2B-blue" alt="Java 20"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

DeepJ is a small Java library for learning, testing, and experimenting with tensors and decoder-only Transformers. It includes automatic differentiation (autograd), five trainable model families, BPE tokenization, model persistence, training utilities, and optional Apple Metal acceleration with a custom implementation of FlashAttention.

DeepJ is an alpha project. Its compact model families are designed to make different Transformer architectures easier to inspect and understand.

## Install

DeepJ requires JDK 20 or newer. Version `0.12.0-alpha` is a GitHub prerelease;
Maven Central publication is pending. Until then, [build from source](#build-and-release).

```xml
<dependency>
    <groupId>io.github.kirstenali</groupId>
    <artifactId>deepj</artifactId>
    <version>0.12.0-alpha</version>
</dependency>
```

API documentation is available in the [Javadoc](https://kirstenali.github.io/DeepJ/api/).

## What is included

- An [N-dimensional `Tensor` API](docs/tensors.md) with first-order [autograd](docs/autograd.md) on CPU and Metal.
- FLOAT32 by default, with optional FLOAT16 and BFLOAT16 storage.
- CPU execution and an optional Metal backend with fused training operations and custom FlashAttention kernels with FP32 accumulation for supported devices and shapes.
- Trainable decoder-only Transformer architectures, described in [Model architectures](#model-architectures).
- Byte and BPE tokenizers, including BPE training and persistence.
- Causal language-model training, AdamW, gradient clipping, and complete resumable checkpoints.
- [KV-cached text generation](docs/inference.md) on CPU and Metal.
- File-backed dataset loaders for sequential reading, random sampling and response-only fine-tuning.
- Export model bundles with weights, configuration, tokenizer files, and a model card for sharing on Hugging Face.
- An optional JavaFX desktop chat UI and streaming CLI.

## Desktop chat

![DeepJ Chat desktop interface](docs/chat-ui.png)

The optional desktop UI loads a local DeepJ checkpoint and streams generated text as it arrives. It includes selectable themes and a **See alternative responses** button for the same prompt.
It automatically uses Metal on supported Apple GPUs and falls back to the CPU elsewhere.

Under Settings → Generation settings:

| Setting | What it changes |
|---|---|
| Temperature | Lower values favor likely next tokens; higher values give other choices a better chance. |
| Top-k | Samples from the top `k` token choices; `0` considers all. |
| Max tokens | Limits how many new tokens are generated; they are not words. |
| Seed | Controls sampling randomness; changing it can change the response. |
| Repetition penalty | Makes the model less likely to repeat itself. `1.0` turns it off; try `1.1` if responses loop. |
| Rolling KV cache | Keeps decoding incremental after the window fills for RoPE models. Off by default; long responses may change. |

Run it from this repository with JDK 20 or newer:

```bash
JAVA_HOME=$(/usr/libexec/java_home -v 20) \
  mvn compile org.openjfx:javafx-maven-plugin:0.0.8:run
```

Choose a `.dj` model file in the app. The screenshot shows the initial screen before a model is loaded.

## Command line

![DeepJ CLI with its coloured wordmark and streaming terminal chat](docs/cli.png)

Build the optional terminal app with JDK 20 or newer:

```bash
mvn package -Pcli
java --enable-native-access=ALL-UNNAMED -jar target/deepj-cli.jar chat ./frisky-90m
```

Chat with streaming responses and selectable themes, generate text for scripts, inspect checkpoints,
or check backend availability. Prompts use the same saved format as the desktop UI, including
Frisky's instruction format. Add `--backend cpu` for CPU-only execution.
See the [CLI guide](docs/cli.md) for commands and controls.

## Quick start

Download a DeepJ model bundle containing `model.dj` and `tokenizer.bpe`, then load it from one directory:

```java
import io.github.kirstenali.deepj.models.DeepJChat;

import java.nio.file.Path;

DeepJChat chat = DeepJChat.load(Path.of("downloaded-model"));
String answer = chat.generate("What is a computer?");
```

The loader reads the tokenizer and checkpoint settings, applies the saved prompt format,
and returns only the generated answer. `chat.stream(prompt, listener)` delivers partial
answers as they are generated. Each call is independent; conversation history is not
retained. `ModelLoader` and `TextGenerator` APIs remain available for advanced use.

## Model architectures

DeepJ provides the following decoder-only Transformer architectures:

| Model | Main components |
|---|---|
| `DeepJOrigin` | Causal multi-head attention, learned positional embeddings, LayerNorm, GELU MLP |
| `DeepJOrbit` | Causal multi-head attention, RoPE, RMSNorm, SwiGLU MLP |
| `DeepJPrism` | Low-rank Q/KV attention, RoPE, RMSNorm, SwiGLU MLP |
| `DeepJNova` | Grouped-query attention, RoPE, RMSNorm, bias-free SwiGLU MLP; optional QK normalization and tied input/output embeddings |
| `DeepJLoop` | Grouped-query attention, RoPE, RMSNorm, bias-free SwiGLU MLP; repeated shared core, context reinjection and loop-aware residual scaling |

## Frisky 90M

Meet [Frisky 90M](https://huggingface.co/netsrik/frisky-90m), a 90-million-parameter DeepJ Prism model trained from scratch through pretraining, mid-training, and instruction fine-tuning.

Download its `model.dj` and `tokenizer.bpe` files to try the quick start above or chat in the desktop UI or CLI. Although Frisky can produce coherent responses, it is a small experimental model and can repeat itself or make factual, reasoning, maths and coding errors. The [training examples](src/main/java/io/github/kirstenali/deepj/examples) remain available for exploration; their prepared datasets are not included in the repository.

## Current limitations

- The public API and checkpoint format may change before a stable release.
- Metal support is limited to Apple Silicon macOS; other platforms use CPU execution.
- Default generation rebuilds the KV cache after each context-window shift. Optional [rolling generation](docs/inference.md#rolling-generation) avoids this for RoPE models, with different behaviour after eviction.
- Hugging Face bundles require DeepJ and cannot be loaded directly by Python Transformers.
- The library has not been validated for safety-critical or production inference workloads.

## Build and release

Build the library locally:

```bash
JAVA_HOME=$(/usr/libexec/java_home -v 20) mvn clean verify
```

See [coding standards and regression testing](docs/quality-and-regression.md) for the enforced checks and validation requirements.

Maintainers can create signed source, Javadoc, and binary artifacts and publish them through the Central Portal with:

```bash
JAVA_HOME=$(/usr/libexec/java_home -v 20) \
  mvn clean deploy -Pcentral-release
```

Maven Central releases are immutable; update the project version before publishing another release.

## License

DeepJ is available under the [MIT License](LICENSE).
