# Desktop app

The DeepJ desktop app supports streamed chat, Qwen Thinking and Agent mode,
image generation, video playback, and saving generated media.

## Download and launch

Download the [desktop JAR for Apple Silicon macOS](https://github.com/KirstenAli/DeepJ/releases/download/v1.0.0/deepj-desktop.jar).
It includes JavaFX and the app's runtime dependencies. Install an Apple Silicon
JDK 20 or newer, place the JAR in a convenient directory, and run:

```bash
java -Xmx3g --enable-native-access=ALL-UNNAMED -jar deepj-desktop.jar
```

The downloaded JAR contains macOS ARM64 native libraries. Other operating systems
and architectures can build a desktop JAR on their own machine using the source
instructions below. Model weights are downloaded separately.

## Choose a model

Use the model menu in the upper left:

| Model | Files to select | Guide |
| --- | --- | --- |
| Qwen3-1.7B | `config.json` beside the tokenizer and weight shards | [Qwen3](qwen3.md) |
| Gemma 2 | `config.json` beside the tokenizer and weight shards | [Gemma 2](gemma2.md) |
| Frisky 90M | `model.dj` with `tokenizer.bpe` in the same directory | [Language models](language-models.md#frisky-90m) |
| SANA-Sprint | **Open image model folder…**, then the full pipeline directory | [Image generation](sana-sprint.md) |
| SANA-Video | **Open video model folder…**, then the full pipeline directory | [Video generation](sana-video.md) |

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

The profile packages the host's JavaFX native libraries. To produce the CLI and
desktop JARs together, use `mvn package -Pcli,desktop`.
The [CLI JAR](https://github.com/KirstenAli/DeepJ/releases/download/v1.0.0/deepj-cli.jar)
and [CLI guide](cli.md) remain available for terminal chat.
