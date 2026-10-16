<p align="center">
  <img src="./deepj_logo.svg" alt="DeepJ logo" width="170">
</p>

# DeepJ

<p align="center">
  <a href="https://central.sonatype.com/artifact/io.github.kirstenali/deepj"><img src="https://img.shields.io/maven-central/v/io.github.kirstenali/deepj.svg?label=Maven%20Central" alt="Maven Central"></a>
  <a href="https://adoptium.net/"><img src="https://img.shields.io/badge/Java-20%2B-blue" alt="Java 20"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

DeepJ is a small Java library for learning, testing, and experimenting with tensors and decoder-only Transformers. It includes built-in gradient calculations, three trainable model families, BPE tokenization, model persistence, training utilities, and optional Apple Metal acceleration.

DeepJ is an alpha project. Its compact model families are designed to make different Transformer architectures easier to inspect and understand.

## Install

DeepJ requires JDK 20 or newer.

```xml
<dependency>
    <groupId>io.github.kirstenali</groupId>
    <artifactId>deepj</artifactId>
    <version>0.9.5-alpha</version>
</dependency>
```

API documentation is available in the [Javadoc](https://kirstenali.github.io/DeepJ/api/).

## What is included

- A two-dimensional `Tensor` API and built-in gradient calculations for model training.
- CPU execution and an optional Metal backend with fused Apple GPU training operations.
- DeepJ Origin with learned positions, LayerNorm, and GELU.
- DeepJ Orbit with RoPE, RMSNorm, and SwiGLU.
- DeepJ Prism with low-rank Q/KV attention, RoPE, RMSNorm, and SwiGLU.
- Byte and BPE tokenizers, including BPE training and persistence.
- Causal language-model training, AdamW, gradient clipping, and complete resumable checkpoints.
- Bounded-memory sequential, random-access, and response-only datasets.
- Versioned model bundles for sharing DeepJ checkpoints on Hugging Face.
- A small optional JavaFX chat UI.

## Desktop chat

The optional desktop UI loads a local DeepJ checkpoint and streams generated text as it arrives. It includes selectable themes and keeps generation settings tucked away.
It automatically uses Metal on supported Apple GPUs and falls back to the CPU elsewhere.

![DeepJ Chat desktop interface](docs/chat-ui.png)

Run it from this repository with JDK 20 or newer:

```bash
JAVA_HOME=$(/usr/libexec/java_home -v 20) \
  mvn org.openjfx:javafx-maven-plugin:0.0.8:run
```

Choose a `.dj` model file in the app. The screenshot shows the initial screen before a model is loaded.

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
retained. Existing `ModelLoader` and `TextGenerator` APIs remain available for advanced
use. Older checkpoints still require an explicitly constructed model.

## Models

| Model | Main components |
|---|---|
| `DeepJOrigin` | Learned positions, causal multi-head attention, LayerNorm, GELU MLP |
| `DeepJOrbit` | RoPE attention, RMSNorm, SwiGLU MLP |
| `DeepJPrism` | Low-rank Q/KV attention, RoPE, RMSNorm, SwiGLU MLP |

All three models share DeepJ's causal language-model API. They currently recalculate the full context for every generated token because attention caching is not yet implemented.

## Frisky 90M

[Frisky 90M](https://huggingface.co/netsrik/frisky-90m) is a 90,128,896-parameter DeepJ Prism model trained from scratch with DeepJ. Download its `model.dj` and `tokenizer.bpe` files together, then use the quick-start example above. The checkpoint stores its architecture and instruction-response prompt format, so neither needs to be entered manually. Its files use DeepJ formats, not PyTorch or Python Transformers formats.

The repository includes the three staged training examples: `TrainDeepJ90M`, `MidTrainDeepJ90M`, and `FineTuneDeepJ90M`. Base pretraining used FineWeb-Edu; the later stages used a mixture derived from SmolTalk, MMLU, GSM8K, and ARC. Training requires those separately prepared datasets, which are not included in this repository. The pipeline uses bounded-memory datasets and complete checkpoints preserving weights, Adam state, progress, and dataset position.

```bash
export DEEPJ_JDK=$(/usr/libexec/java_home -v 20)
JAVA_HOME="$DEEPJ_JDK" mvn compile
```

`deepj.batchSize` is the number of sequences averaged before each optimizer update. DeepJ processes those sequences one at a time, accumulates their gradients, then averages and clips the result.

Measure pretraining progress against the same held-out FineWeb-Edu examples after each checkpoint:

```bash
"$DEEPJ_JDK/bin/java" \
  -Xmx6g \
  --enable-native-access=ALL-UNNAMED \
  -cp target/classes \
  io.github.kirstenali.deepj.examples.EvaluateDeepJ90MPretraining
```

Export the final instruction-tuned checkpoint as a Hugging Face-ready bundle:

```bash
"$DEEPJ_JDK/bin/java" \
  -Xmx6g \
  -cp target/classes \
  io.github.kirstenali.deepj.examples.ExportFrisky90M
```

The exporter removes local training paths from the public checkpoint. Set `deepj.checkpoint` and `deepj.bundle` to override its default input and output paths.

## Metal acceleration

On Apple Silicon macOS, enable Metal when it is available:

```java
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;

if (MetalBackend.isAvailable()) {
    Tensor.setBackend(new MetalBackend());
}
```

The CPU backend remains the portable default. Rebuild the bundled native library after editing Metal or JNI code:

```bash
./native/build-macos.sh
```

## Verify the project

Run the standard tests and code checks with JDK 20:

```bash
JAVA_HOME=$(/usr/libexec/java_home -v 20) mvn clean verify
```

Maven installs a local Node runtime and checks the chat HTML, CSS, and JavaScript for formatting, errors, short functions, and duplicate code. JavaFX window tests need a desktop session and can be run separately:

```bash
JAVA_HOME=$(/usr/libexec/java_home -v 20) \
  mvn test -Dtest=ChatWindowTest -Ddeepj.uiTests=true
```

The numerical tests compare gradients calculated by the model layers with estimates from small input changes. On supported Apple hardware, `MetalBackendDifferentialTest` also compares CPU and Metal forward passes, backward passes, and parameter gradients:

```bash
JAVA_HOME=$(/usr/libexec/java_home -v 20) \
  mvn test -Dtest=MetalBackendDifferentialTest
```

## Current limitations

- The public API and checkpoint format may change before a stable release.
- Tensors are currently two-dimensional and use `float32` values.
- Training uses backward calculations built into each layer rather than a general-purpose automatic differentiation engine.
- Metal support is limited to Apple Silicon macOS; other platforms use CPU execution.
- Text generation recalculates the full context for every new token because attention caching is not yet implemented.
- Hugging Face bundles require DeepJ and cannot be loaded directly by Python Transformers.
- The library has not been validated for safety-critical or production inference workloads.

## Build and release

Build the library locally:

```bash
JAVA_HOME=$(/usr/libexec/java_home -v 20) mvn clean verify
```

Maintainers can create signed source, Javadoc, and binary artifacts and publish them through the Central Portal with:

```bash
JAVA_HOME=$(/usr/libexec/java_home -v 20) \
  mvn clean deploy -Pcentral-release
```

Maven Central releases are immutable; update the project version before publishing another release.

## License

DeepJ is available under the [MIT License](LICENSE).
