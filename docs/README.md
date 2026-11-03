# DeepJ documentation

DeepJ is a Java neural network library built on N-dimensional tensors, automatic
differentiation, composable layers, and shared CPU/Metal execution.

[Project overview](../README.md) · [Website](https://kirstenali.github.io/DeepJ/) ·
[API reference](https://kirstenali.github.io/DeepJ/api/)

## Training

| Guide | Start here for |
|---|---|
| [Network training](training.md) | Training APIs by network family, minibatches, losses, AdamW, and training/evaluation mode |
| [Language-model training](training.md#causal-language-model-training) | Token batches, response masks, gradient clipping, and accumulation |
| [Distributed training](../distributed/README.md) | Optional data-parallel training across CPU or Apple Silicon workers, data shards, and checkpoints |

## Core library

| Guide | Start here for |
|---|---|
| [Tensors and precision](tensors.md) | Shapes, views, broadcasting, dtypes, and storage |
| [Automatic differentiation](autograd.md) | Gradient recording, backward, mutation checks, and graph lifetime |
| [Networks and layers](neural-networks.md) | MLPs, CNNs, recurrent networks, autoencoders, Transformer encoders, and composition |
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

These guides and the generated API reference document DeepJ `0.13.0-alpha`.
See the [release notes](releases/0.13.0-alpha.md) and
[tagged source](https://github.com/KirstenAli/DeepJ/tree/v0.13.0-alpha) for this version.

## Development and verification

- [Tests and coverage](testing.md): portable and full Metal/desktop build profiles.
- [Coding and regression standards](quality-and-regression.md): code quality, numerical
  references, state restoration, performance, and memory qualification.

Builds require JDK 20 or newer. See [installation and source builds](../README.md#install).
