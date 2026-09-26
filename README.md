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
    <version>0.8.0-alpha</version>
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

## Quick start

Tensor operations use the CPU backend by default:

```java
import io.github.kirstenali.deepj.tensor.Tensor;

Tensor inputs = Tensor.from2D(new float[][]{
        {1.0f, 2.0f},
        {3.0f, 4.0f}
});
Tensor weights = Tensor.from2D(new float[][]{
        {0.5f, 1.0f},
        {-1.0f, 2.0f}
});

Tensor output = inputs.matmul(weights).reluActivation();
```

Load a DeepJ Prism checkpoint with its BPE tokenizer:

```java
import io.github.kirstenali.deepj.models.TextGenerator;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.models.prism.DeepJPrism;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModel;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETokenizer;

import java.nio.file.Path;

Path directory = Path.of("downloaded-model");
BPEModel bpe = BPEModelIO.load(directory.resolve("tokenizer.bpe"));
BPETokenizer tokenizer = new BPETokenizer(bpe);
DeepJPrismConfig config = new DeepJPrismConfig(
        tokenizer.vocabSize(), 128, 128, 4, 4, 384, 64, 32);
DeepJPrism model = new DeepJPrism(config, 42L);

model.load(directory.resolve("model.dj"));
String text = TextGenerator.generate(
        model, tokenizer, config, "Once upon a time", 80, 0.8f, 40, 2026L);
```

The model configuration must match the saved checkpoint.

## Models

| Model | Main components |
|---|---|
| `DeepJOrigin` | Learned positions, causal multi-head attention, LayerNorm, GELU MLP |
| `DeepJOrbit` | RoPE attention, RMSNorm, SwiGLU MLP |
| `DeepJPrism` | Low-rank Q/KV attention, RoPE, RMSNorm, SwiGLU MLP |

All three models share DeepJ's causal language-model API. They currently recalculate the full context for every generated token because attention caching is not yet implemented.

## Train on TinyStories

The repository contains a bounded-memory training example for `sample_data/TinyStories-train.txt`. It trains a BPE tokenizer, writes rolling checkpoints, and saves final weights.

```bash
export DEEPJ_JDK=$(/usr/libexec/java_home -v 20)
JAVA_HOME="$DEEPJ_JDK" mvn compile

"$DEEPJ_JDK/bin/java" \
  -Ddeepj.steps=10000 \
  -cp target/classes \
  io.github.kirstenali.deepj.examples.TrainDeepJPrismTinyStories
```

Useful overrides include `deepj.batchSize`, `deepj.seqLen`, `deepj.dModel`, `deepj.layers`, `deepj.vocabSize`, `deepj.learningRate`, `deepj.output`, and `deepj.checkpointEvery`.

`deepj.batchSize` is the number of sequences averaged before each optimizer update. DeepJ processes those sequences one at a time, accumulates their gradients, then averages and clips the result.

Evaluate the saved model on the validation split:

```bash
"$DEEPJ_JDK/bin/java" \
  -Ddeepj.evalCorpus=sample_data/TinyStories-valid.txt \
  -cp target/classes \
  io.github.kirstenali.deepj.examples.EvaluateDeepJPrismTinyStories
```

Export a Hugging Face-ready DeepJ bundle:

```bash
"$DEEPJ_JDK/bin/java" \
  -cp target/classes \
  io.github.kirstenali.deepj.examples.ExportDeepJPrismTinyStories
```

The published demonstration model is [netsrik/deepj-tinystories](https://huggingface.co/netsrik/deepj-tinystories). Its `model.dj` and `tokenizer.bpe` files use DeepJ formats; they are not PyTorch or Transformers checkpoints.

## Train the 90M example

DeepJ also includes a staged 90M-parameter training pipeline using FineWeb-Edu, SmolTalk, MMLU, GSM8K, and ARC. It uses bounded-memory datasets and complete checkpoints that preserve model weights, Adam state, training progress, and dataset position.

Measure pretraining progress against the same held-out FineWeb-Edu examples after each checkpoint:

```bash
"$DEEPJ_JDK/bin/java" \
  -Xmx6g \
  --enable-native-access=ALL-UNNAMED \
  -cp target/classes \
  io.github.kirstenali.deepj.examples.EvaluateDeepJ90MPretraining
```

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

Run the complete test and style suite with JDK 20:

```bash
JAVA_HOME=$(/usr/libexec/java_home -v 20) mvn clean test
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
