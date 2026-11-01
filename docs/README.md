# DeepJ documentation

DeepJ is a Java neural network library built on N-dimensional tensors, automatic
differentiation, composable layers, and shared CPU/Metal execution.

[Project overview](../README.md) · [Website](https://kirstenali.github.io/DeepJ/) ·
[API reference](https://kirstenali.github.io/DeepJ/api/)

## Core library

| Guide | Start here for |
|---|---|
| [Tensors and precision](tensors.md) | Shapes, views, broadcasting, dtypes, and storage |
| [Automatic differentiation](autograd.md) | Gradient recording, backward, mutation checks, and graph lifetime |
| [Networks and layers](neural-networks.md) | MLPs, CNNs, recurrent networks, autoencoders, Transformer encoders, and composition |
| [Training](training.md) | Supervised batching, losses, optimizers, and training/evaluation mode |
| [Saving and loading](persistence.md) | Parameters, persistent buffers, model files, and resumable state |
| [CPU and Metal](backends.md) | Backend selection, GPU execution, precision, and temporary tensors |

## Language models

| Guide | Covers |
|---|---|
| [Language models and tools](language-models.md) | Decoder families, tokenizers, pretrained models, desktop chat, and CLI |
| [Incremental generation](inference.md) | KV caches, sessions, context windows, and rolling generation |
| [Looped training](looped-training.md) | Shared decoder layers and loop-specific training |
| [CLI reference](cli.md) | Terminal commands, settings, themes, and scripting |

## Availability

These guides and the generated API reference describe the prepared `0.13.0-alpha`
release, including `io.github.kirstenali.deepj.nn`. Maven Central publication is pending;
build this source checkout to use those APIs until the release is published.

The currently published `0.12.0-alpha` artifact predates `nn`. Use its
[release notes](releases/0.12.0-alpha.md) and
[tagged source](https://github.com/KirstenAli/DeepJ/tree/v0.12.0-alpha) when working with
that version. The [0.13.0-alpha release notes](releases/0.13.0-alpha.md) describe the
new network APIs and the release preparation status.

## Development and verification

- [Tests and coverage](testing.md): portable and full Metal/desktop build profiles.
- [Coding and regression standards](quality-and-regression.md): code quality, numerical
  references, state restoration, performance, and memory qualification.

Builds require JDK 20 or newer. See [installation and source builds](../README.md#install).
