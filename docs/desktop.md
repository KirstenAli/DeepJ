# Desktop app

The DeepJ desktop app supports streamed chat, Qwen Thinking and Agent mode,
image generation, video playback, and saving generated media.

## Download and launch

Download the [desktop JAR](https://github.com/KirstenAli/DeepJ/releases/download/v1.0.0/deepj-desktop.jar).
It includes JavaFX and the app's runtime dependencies. Install a matching
JDK 20 or newer, place the JAR in a convenient directory, and run:

```bash
java -Xmx3g --enable-native-access=ALL-UNNAMED -jar deepj-desktop.jar
```

The same JAR includes JavaFX for Apple Silicon macOS and Windows x64; JavaFX
selects the matching libraries automatically. Auto uses Metal on supported Macs
and CPU on Windows. Model weights are downloaded separately.

## CPU or Metal

Pass `--backend cpu`, `--backend metal`, or `--backend auto` after the JAR name.
Without an argument, Auto uses Metal when available, otherwise CPU. The startup
choice applies to chat, images, and videos.
Requesting Metal on an unsupported computer reports an error.

For Qwen3-1.7B on CPU, allow room for weights and working buffers in Java's heap:

```bash
java -Xmx6g --enable-native-access=ALL-UNNAMED -jar deepj-desktop.jar --backend cpu
```

CPU heap starting points for the bundled model examples:

| Model | CPU heap |
| --- | --- |
| Frisky 90M | `-Xmx3g` |
| Qwen3-1.7B, BF16 | `-Xmx6g` |
| Gemma 2 2B-IT, BF16 | `-Xmx8g` |
| SANA-Sprint | `-Xmx8g` |
| SANA-Video, short preview | `-Xmx12g` |

Image and video memory use grows with resolution, clip length, and precision.
These are starting budgets, not limits on total process memory. The video worker
inherits the CPU heap limit; its Metal heap budget remains separate.

For Metal, use `--backend metal`. From source, pass
`-Djavafx.args="--backend cpu"` to `mvn compile javafx:run`.

## Choose a model

Choose **Open local model…** from the model menu in the upper left. This opens
a file picker; select the file listed for your model:

| Model | Files to select | Guide |
| --- | --- | --- |
| Qwen3-1.7B | `config.json` beside the tokenizer and weight shards | [Qwen3](qwen3.md) |
| Gemma 2 | `config.json` beside the tokenizer and weight shards | [Gemma 2](gemma2.md) |
| Frisky 90M | `model.dj` with `tokenizer.bpe` in the same directory | [Language models](language-models.md#frisky-90m) |
| SANA-Sprint | `model_index.json` in the full pipeline's root directory | [Image generation](sana-sprint.md) |
| SANA-Video | `model_index.json` in the full pipeline's root directory | [Video generation](sana-video.md) |

For images and videos, select the pipeline's `model_index.json` once. DeepJ reads
the component configuration files automatically.

Qwen defaults to the checkpoint's precision for weights and activations, normally
BF16. Open **Chat options** for Thinking, Agent mode, and conversation history.
History starts off. Model-specific guides describe the available controls and
generation settings. Video generation runs in its separate worker with its own
memory budget.

## Build from source

Build on the operating system and architecture where the app will run:

```bash
mvn package -Pdesktop
java -Xmx3g --enable-native-access=ALL-UNNAMED -jar target/deepj-desktop.jar
```

The profile packages the host's JavaFX libraries. To build the combined Mac and
Windows download, use `mvn package -Pdesktop,desktop-mac-windows`. Add `cli` to
the profile list to build the CLI JAR as well.
The [CLI JAR](https://github.com/KirstenAli/DeepJ/releases/download/v1.0.0/deepj-cli.jar)
and [CLI guide](cli.md) remain available for terminal chat.
