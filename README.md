<p align="center">
  <img src="./deepj_logo.svg" alt="DeepJ logo" width="170">
</p>

# DeepJ

<p align="center">
  <a href="https://central.sonatype.com/artifact/io.github.kirstenali/deepj"><img src="https://img.shields.io/maven-central/v/io.github.kirstenali/deepj.svg?label=Maven%20Central" alt="Maven Central"></a>
  <a href="https://adoptium.net/"><img src="https://img.shields.io/badge/Java-20%2B-blue" alt="Java 20"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

DeepJ is a small Java library for learning, testing, and experimenting with tensors and decoder-only Transformers. It includes built-in gradient calculations for model training, GPT-, Llama-, and DeepSeek-style models, BPE tokenization, training utilities, model persistence, and optional Apple Metal acceleration.

DeepJ is an alpha project. Its model implementations are compact educational architectures, not drop-in reproductions of the official GPT, Llama, or DeepSeek releases.

## Install

DeepJ requires JDK 20 or newer.

```xml
<dependency>
    <groupId>io.github.kirstenali</groupId>
    <artifactId>deepj</artifactId>
    <version>0.7.0-alpha</version>
</dependency>
```

API documentation is available in the [Javadoc](https://kirstenali.github.io/DeepJ/api/).

## What is included

- A two-dimensional `Tensor` API and built-in gradient calculations for model training.
- CPU execution and an optional Metal backend with fused Apple GPU training operations.
- GPT-style attention with learned positions, LayerNorm, and GELU.
- Llama-style attention with RoPE, RMSNorm, and SwiGLU.
- DeepSeek-inspired low-rank Q/KV attention with RoPE, RMSNorm, and SwiGLU.
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

Create a compact DeepSeek-style model and generate from trained or loaded weights:

```java
import io.github.kirstenali.deepj.models.TextGenerator;
import io.github.kirstenali.deepj.models.deepseek.DeepSeekConfig;
import io.github.kirstenali.deepj.models.deepseek.DeepSeekModel;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.nio.file.Path;

Tokenizer tokenizer = new ByteTokenizer();
DeepSeekConfig config = new DeepSeekConfig(
        tokenizer.vocabSize(), 128, 64, 4, 2, 256, 32, 16);
DeepSeekModel model = new DeepSeekModel(config, 42L);

model.load(Path.of("model.dj"));
String text = TextGenerator.generate(
        model, tokenizer, config, "Once upon a time", 80, 0.8f, 20, 42L);
```

The model configuration must match the saved checkpoint.

## Models

| Model | Main components |
|---|---|
| `GPTModel` | Learned positions, causal multi-head attention, LayerNorm, GELU MLP |
| `LlamaModel` | RoPE attention, RMSNorm, SwiGLU MLP |
| `DeepSeekModel` | Compact low-rank Q/KV attention, RoPE, RMSNorm, SwiGLU MLP |

These models currently recompute the context for each generated token; incremental KV caching is not yet implemented. The DeepSeek-style model is inspired by Multi-Head Latent Attention but is not an exact DeepSeek-V2, V3, or R1 implementation.

## Train on TinyStories

The repository contains a bounded-memory training example for `sample_data/TinyStories-train.txt`. It trains a BPE tokenizer, writes rolling checkpoints, and saves final weights.

```bash
export DEEPJ_JDK=$(/usr/libexec/java_home -v 20)
JAVA_HOME="$DEEPJ_JDK" mvn compile

"$DEEPJ_JDK/bin/java" \
  -Ddeepj.steps=10000 \
  -cp target/classes \
  io.github.kirstenali.deepj.examples.TrainDeepSeekTinyStories
```

Useful overrides include `deepj.batchSize`, `deepj.seqLen`, `deepj.dModel`, `deepj.layers`, `deepj.vocabSize`, `deepj.learningRate`, `deepj.output`, and `deepj.checkpointEvery`.

Evaluate the saved model on the validation split:

```bash
"$DEEPJ_JDK/bin/java" \
  -Ddeepj.evalCorpus=sample_data/TinyStories-valid.txt \
  -cp target/classes \
  io.github.kirstenali.deepj.examples.EvaluateDeepSeekTinyStories
```

Export a Hugging Face-ready DeepJ bundle:

```bash
"$DEEPJ_JDK/bin/java" \
  -cp target/classes \
  io.github.kirstenali.deepj.examples.ExportDeepSeekTinyStories
```

The published demonstration model is [netsrik/deepj-tinystories](https://huggingface.co/netsrik/deepj-tinystories). Its `model.dj` and `tokenizer.bpe` files use DeepJ formats; they are not PyTorch or Transformers checkpoints.

## Train the 90M example

DeepJ also includes a staged 90M-parameter training pipeline using FineWeb-Edu, SmolTalk, MMLU, GSM8K, and ARC. It uses bounded-memory datasets and complete checkpoints that preserve model weights, Adam state, training progress, and dataset position.

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
- Each model layer has its own gradient code. DeepJ does not calculate gradients automatically for every tensor operation.
- Metal support is limited to Apple Silicon macOS; other platforms use CPU execution.
- Generation does not yet use a KV cache.
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
