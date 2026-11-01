<p align="center">
  <img src="./deepj_logo.svg" alt="DeepJ logo" width="170">
</p>

# DeepJ

<p align="center">
  <a href="https://central.sonatype.com/artifact/io.github.kirstenali/deepj"><img src="https://img.shields.io/maven-central/v/io.github.kirstenali/deepj.svg?label=Maven%20Central" alt="Maven Central"></a>
  <a href="https://adoptium.net/"><img src="https://img.shields.io/badge/Java-20%2B-blue" alt="Java 20"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

DeepJ is a Java library for building, training, and running neural networks. It provides N-dimensional tensors, automatic differentiation, composable layers, and CPU execution with optional Apple Metal acceleration.

Build feed-forward networks, convolutional networks, recurrent networks, autoencoders, and Transformers on the same tensor and autograd foundation.

[Website](https://kirstenali.github.io/DeepJ/) · [Documentation](docs/README.md) · [API reference](https://kirstenali.github.io/DeepJ/api/)

## Supported networks

| Family | Implementations |
|---|---|
| Feed-forward networks | `FNN`, with configurable hidden and output activations |
| Convolutional networks | `Conv1D`, `Conv2D`, max/average pooling, and global average pooling |
| Recurrent networks | `RNN`, `GRU`, `LSTM`, individual cells, explicit state, and per-sample sequence lengths |
| Autoencoders | Composable `Autoencoder` with separate encoder and decoder access |
| Transformer encoders | Bidirectional attention, padding masks, and masked mean pooling |
| Decoder Transformers | DeepJ Origin, Orbit, Prism, Nova, and Loop for causal language modeling |

## Install

DeepJ requires JDK 20 or newer.

Add DeepJ to your project:

```xml
<dependency>
    <groupId>io.github.kirstenali</groupId>
    <artifactId>deepj</artifactId>
    <version>0.13.0-alpha</version>
</dependency>
```

## Try Frisky 90M

[Frisky 90M](https://huggingface.co/netsrik/frisky-90m) is a 90-million-parameter language model trained from scratch with DeepJ. Download it and chat locally with the DeepJ CLI.

Download [`model.dj`](https://huggingface.co/netsrik/frisky-90m/resolve/main/model.dj?download=true) and [`tokenizer.bpe`](https://huggingface.co/netsrik/frisky-90m/resolve/main/tokenizer.bpe?download=true) into a folder named `frisky-90m`. Place the [CLI JAR](https://github.com/KirstenAli/DeepJ/releases/download/v0.13.0-alpha/deepj-cli.jar) alongside that folder, then run:

```bash
java --enable-native-access=ALL-UNNAMED -jar deepj-cli.jar chat ./frisky-90m
```

The CLI streams responses, supports selectable themes, and uses Metal when available with CPU fallback. See the [CLI guide](docs/cli.md) for commands and controls, and the [model card](https://huggingface.co/netsrik/frisky-90m) for training details.

![DeepJ CLI streaming a response from Frisky 90M](docs/cli.png)

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
| <a id="desktop-chat"></a>[Language models](docs/language-models.md) | Decoder architectures, tokenization, text generation, chat, and model bundles |

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
