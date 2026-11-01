# Language models and tools

DeepJ's language-model APIs build on the same tensors, autograd, layers, and CPU/Metal
backends as the other network families. This guide covers decoder Transformers,
pretrained checkpoints, text generation, and the optional chat interfaces.

See the [documentation index](README.md) for feed-forward, convolutional, recurrent,
autoencoder, and encoder-Transformer APIs.

## Model architectures

DeepJ provides the following decoder-only Transformer architectures:

Complete decoder models, configurations, loading, and generation APIs live under
`io.github.kirstenali.deepj.nn.transformer`. Each family has its own subpackage:
`origin`, `orbit`, `prism`, `nova`, or `loop`. Attention, normalization, and block
implementations remain under `io.github.kirstenali.deepj.layers.transformer`.

When upgrading from earlier builds, replace the `io.github.kirstenali.deepj.models`
package prefix with `io.github.kirstenali.deepj.nn.transformer` and recompile.
The model constructors, calculations, checkpoint formats, and CLI commands are unchanged.

| Model | Main components |
|---|---|
| `DeepJOrigin` | Causal multi-head attention, learned positional embeddings, LayerNorm, GELU MLP |
| `DeepJOrbit` | Causal multi-head attention, RoPE, RMSNorm, SwiGLU MLP |
| `DeepJPrism` | Low-rank Q/KV attention, RoPE, RMSNorm, SwiGLU MLP |
| `DeepJNova` | Grouped-query attention, RoPE, RMSNorm, bias-free SwiGLU MLP; optional QK normalization and tied input/output embeddings |
| `DeepJLoop` | Grouped-query attention, RoPE, RMSNorm, bias-free SwiGLU MLP; repeated shared core, context reinjection and loop-aware residual scaling |

## Load a language model

Download a DeepJ model bundle containing `model.dj` and `tokenizer.bpe`, then load it from one directory:

```java
import io.github.kirstenali.deepj.nn.transformer.DeepJChat;

import java.nio.file.Path;

DeepJChat chat = DeepJChat.load(Path.of("downloaded-model"));
String answer = chat.generate("What is a computer?");
```

The loader reads the tokenizer and checkpoint settings, applies the saved prompt format,
and returns only the generated answer. `chat.stream(prompt, listener)` delivers partial
answers as they are generated. Each call is independent; conversation history is not
retained. `ModelLoader` and `TextGenerator` APIs remain available for advanced use.

## Desktop chat

![DeepJ Chat desktop interface](chat-ui.png)

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

![DeepJ CLI with its coloured wordmark and streaming terminal chat](cli.png)

Build the optional terminal app with JDK 20 or newer:

```bash
mvn package -Pcli
java --enable-native-access=ALL-UNNAMED -jar target/deepj-cli.jar chat ./frisky-90m
```

Chat with streaming responses, inspect checkpoints, or check backend availability.
Choose from several themes to customise the interface. Add `--backend cpu` for CPU-only execution.
See the [CLI guide](cli.md) for commands and controls.

## Frisky 90M

Meet [Frisky 90M](https://huggingface.co/netsrik/frisky-90m), a 90-million-parameter DeepJ Prism model trained from scratch through pretraining, mid-training, and instruction fine-tuning.

Download its `model.dj` and `tokenizer.bpe` files to try the model-loading section above or chat in the desktop UI or CLI. Although Frisky can produce coherent responses, it is a small experimental model and can repeat itself or make factual, reasoning, maths and coding errors. The [training examples](../src/main/java/io/github/kirstenali/deepj/examples) remain available for exploration; their prepared datasets are not included in the repository.

## Text data and persistence

Byte and trainable BPE tokenizers support language-model training and generation.
Text dataset loaders provide sequential reading, random sampling, and response-only
fine-tuning. Causal training includes cross entropy, AdamW, gradient clipping, and
resumable checkpoints.

A model bundle can contain weights, configuration, tokenizer files, and a model card
for sharing on Hugging Face. These bundles use DeepJ's own format. See
[saving and loading](persistence.md) for the distinction between network weight
archives, model checkpoints, and complete training state.

## Language-model limitations

- Default generation rebuilds the KV cache after each context-window shift.
  Optional [rolling generation](inference.md#rolling-generation) avoids this for RoPE
  models, with different behavior after eviction.
- Hugging Face bundles require DeepJ and cannot be loaded directly by Python Transformers.
- Tokenization, prompt formatting, sampling, and checkpoint quality affect generated text.
  A library feature or a successful GPU test does not establish a pretrained model's accuracy.

See [incremental generation](inference.md), [looped training](looped-training.md), and
the [CLI reference](cli.md) for the language-model-specific contracts.
