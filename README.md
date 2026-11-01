<p align="center">
  <img src="./deepj_logo.svg" alt="DeepJ logo" width="170">
</p>

# DeepJ

<p align="center">
  <a href="https://central.sonatype.com/artifact/io.github.kirstenali/deepj"><img src="https://img.shields.io/maven-central/v/io.github.kirstenali/deepj.svg?label=Maven%20Central" alt="Maven Central"></a>
  <a href="https://adoptium.net/"><img src="https://img.shields.io/badge/Java-20%2B-blue" alt="Java 20"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

DeepJ is a Java neural network library for building, training, and running models. It provides N-dimensional tensors, automatic differentiation, composable layers, and CPU execution with optional Apple Metal acceleration.

Build feed-forward networks, convolutional networks, recurrent networks, autoencoders, and Transformers on the same tensor and autograd foundation.

[Website](https://kirstenali.github.io/DeepJ/) · [Documentation](docs/README.md) · [API reference](https://kirstenali.github.io/DeepJ/api/)

## Supported networks

| Family | Implementations |
|---|---|
| Feed-forward networks | `MLP`, backed by `FNN`, with configurable hidden and output activations |
| Convolutional networks | `Conv1D`, `Conv2D`, max/average pooling, and global average pooling |
| Recurrent networks | `RNN`, `GRU`, `LSTM`, individual cells, explicit state, and per-sample sequence lengths |
| Autoencoders | Composable `Autoencoder` with separate encoder and decoder access |
| Transformer encoders | Bidirectional attention, padding masks, and masked mean pooling |
| Decoder Transformers | DeepJ Origin, Orbit, Prism, Nova, and Loop for causal language modeling |

The `nn` package APIs are included in the prepared `0.13.0-alpha` release. Maven Central publication is pending; see [installation and availability](#install).

## Install

DeepJ requires JDK 20 or newer.

The prepared release uses these Maven coordinates:

```xml
<dependency>
    <groupId>io.github.kirstenali</groupId>
    <artifactId>deepj</artifactId>
    <version>0.13.0-alpha</version>
</dependency>
```

`0.13.0-alpha` is prepared for release and is not yet published to Maven Central. Until publication, build this source checkout for the `nn` APIs. The currently published `0.12.0-alpha` artifact provides tensors, autograd, `FNN`, and decoder Transformers. See the [0.13.0-alpha release notes](docs/releases/0.13.0-alpha.md) and [availability details](docs/README.md#availability).

## Tensors, gradients, and execution

- One [N-dimensional `Tensor` API](docs/tensors.md) for scalars, vectors, matrices, images, and sequences.
- First-order [automatic differentiation](docs/autograd.md) through tensor operations, including views, broadcasting, and repeated indices.
- FLOAT32 defaults, with optional FLOAT16/BFLOAT16 storage and FLOAT32 gradients.
- A Java CPU backend and optional [Metal execution on Apple Silicon](docs/backends.md). Network arithmetic uses the selected tensor backend.

Convolution, recurrence, and attention all use these shared tensor operations and gradients. See [network implementations and shapes](docs/neural-networks.md) for the API contracts.

## Quick start

Choose a network family from the table above, then compose its layers with `Sequential`. Existing DeepJ layers implement the same `Layer` interface. Activation adapters, flattening, reshaping, dropout, and batch normalization are available alongside the network implementations.

A model's `forward(...)` returns a tensor. Compute a differentiable loss, call `backward()` on that loss, and update `model.parameters()` with an optimizer. [Training](docs/training.md) explains supervised batching, losses, train/evaluation mode, and explicit recurrent state.

Use [network weight archives](docs/persistence.md) to save parameters and persistent buffers, then restore them into the same constructed architecture. Language-model checkpoints and resumable training state have their own documented formats.

## Documentation

Start with the [documentation index](docs/README.md), or go directly to:

| Guide | Covers |
|---|---|
| [Tensors and precision](docs/tensors.md) | Shapes, views, broadcasting, dtypes, and storage |
| [Automatic differentiation](docs/autograd.md) | Gradients, mutation checks, graph lifetime, and custom operations |
| [Networks and layers](docs/neural-networks.md) | Composition, CNNs, recurrence, autoencoders, and encoder Transformers |
| [Training](docs/training.md) | Batching, losses, optimizers, and train/evaluation mode |
| [Saving and loading](docs/persistence.md) | Network weights, persistent buffers, and model checkpoints |
| [CPU and Metal](docs/backends.md) | Backend selection, GPU execution, and temporary tensor ownership |
| [Language models](docs/language-models.md) | Decoder architectures, tokenization, text generation, chat, and model bundles |

The [Javadocs](https://kirstenali.github.io/DeepJ/api/) describe the source used to build them; the package list includes `io.github.kirstenali.deepj.nn` for current-source builds.

## Language-model tools

DeepJ also provides trainable decoder Transformers, byte and BPE tokenizers, KV-cached generation, text dataset loaders, and model bundles. See the [language-model guide](docs/language-models.md) for architecture details and the Frisky 90M checkpoint.

### Desktop chat

The optional JavaFX app loads local language-model checkpoints and streams generated text. See [desktop chat and generation settings](docs/language-models.md#desktop-chat).

### Command line

The optional CLI supports streaming chat, text generation, checkpoint inspection, and backend diagnostics. See the [CLI guide](docs/cli.md).

## Current limitations

- The public API and persistence formats may change before a stable release.
- Metal acceleration requires supported Apple Silicon macOS; other platforms use CPU execution.
- Autograd supports first-order differentiation; backward releases the operation graph.
- Network weight archives contain parameters and persistent buffers, with architecture and training state managed separately.
- Language-model context-window and bundle-format details are documented in the [language-model guide](docs/language-models.md#language-model-limitations).
- The library has not been validated for safety-critical or production inference workloads.

## Build and release

Build and check the current source with JDK 20 or newer:

```bash
mvn clean verify
```

On supported Apple Silicon macOS with a working Metal device and desktop session, run `mvn clean verify -Pfull-coverage` for the hardware-dependent checks. See [coding standards and regression testing](docs/quality-and-regression.md) for the validation requirements.

Maintainers can create signed source, Javadoc, and binary artifacts and publish them through the Central Portal with:

```bash
mvn clean deploy -Pcentral-release
```

Maven Central releases are immutable; update the project version before publishing another release.

## License

DeepJ is available under the [MIT License](LICENSE).
