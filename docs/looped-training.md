# Looped 100M training

DeepJ Loop is a new decoder architecture, not a converted Nova checkpoint.
The example starts with fresh weights and reuses Frisky's BPE tokenizer,
FineWeb-Edu corpus, mid-training mixture, and response-only fine-tuning data.
Existing Frisky and Nova checkpoints are not overwritten.

## Architecture

| Setting | Default |
|---|---|
| Unique trainable parameters | 100,375,680 |
| Vocabulary / context | 16,384 / 1,024 |
| Model / feed-forward width | 576 / 1,664 |
| Query / KV heads | 9 / 3, each 64 dimensions |
| Distinct blocks | 6 prelude + 12 shared core + 6 coda |
| Core passes | 2 |
| Executed block visits | 36 |

The prelude runs once. Each core pass receives both the previous state and
the unchanged prelude representation through shared bias-free projections.
An RMSNorm follows the core on each pass, and the coda runs once at the end.
The context projection is calculated once per forward pass. Embedding and
output weights are tied.

Only weights are shared, not per-pass saved activations. The optimizer updates
each parameter once.
The core residual branches use `1 / (loops * sqrt(coreLayers))` by default;
outer blocks retain their normal residual scale. This is a loop-aware
scaling choice, not a claim to reproduce the entire Loop-μP training recipe.

The design draws on [recurrent-depth transformers](https://arxiv.org/abs/2502.05171),
the [full-versus-truncated BPTT scaling study](https://arxiv.org/abs/2610.26030),
and [Loop-μP](https://arxiv.org/abs/2606.18524).
It reuses DeepJ's existing GQA, RoPE, RMSNorm, SwiGLU and CPU/Metal operations.
On supported Apple GPUs, the default 1,024-token, 64-wide-head configuration
automatically uses FP32 Flash-style attention in each core pass. Other shapes
retain the ordinary attention path.
Looping increases computation and saved-activation memory. It does not guarantee
better reasoning, factual accuracy, or less repetitive text.

## Build and benchmark

Use JDK 20. From the repository root:

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 20)
mvn clean verify

caffeinate -i /usr/bin/time -l "$JAVA_HOME/bin/java" \
  -Xmx8g -XX:+ExitOnOutOfMemoryError \
  --enable-native-access=ALL-UNNAMED \
  -Ddeepj.benchmarkWarmup=5 -Ddeepj.benchmarkSteps=250 \
  -cp target/classes \
  io.github.kirstenali.deepj.examples.BenchmarkDeepJLoop100M
```

The benchmark trains on synthetic tokens, saves nothing, and does not read a
checkpoint or corpus. Stop other GPU workloads before comparing timings.
The ordinary build runs portable tests and skips unavailable hardware tests;
`mvn clean verify -Pfull-coverage` additionally requires a desktop and Metal.

## Training stages

Keep the existing private files under `sample_data/deepj-90m/`.
Pretraining reads only the tokenizer from `checkpoints/deepj-90m/pretrain/`,
not that model's weights. Set `-Ddeepj.base=/path/to/tokenizer-directory`
if the tokenizer is elsewhere.

| Stage | Corpus | Updates | Output directory |
|---|---|---:|---|
| Pretrain | `fineweb-edu.txt` | 2,000,000 | `checkpoints/deepj-loop-100m/pretrain` |
| Midtrain | `midtrain-train.txt` | 256,000 | `checkpoints/deepj-loop-100m/midtrain` |
| Fine-tune | `sft-train.txt` | 22,500 | `checkpoints/deepj-loop-100m/sft` |

These are the current 100M example's budgets. Each update uses one
1,024-token sequence. Pretraining remains sequential, mid-training uses
random-access sampling, and fine-tuning supervises response tokens only.
The held-out FineWeb-Edu range stays excluded from pretraining.

Run each stage after the preceding stage finishes:

```bash
caffeinate -i "$JAVA_HOME/bin/java" -Xmx8g \
  --enable-native-access=ALL-UNNAMED -cp target/classes \
  io.github.kirstenali.deepj.examples.TrainDeepJLoop100M

caffeinate -i "$JAVA_HOME/bin/java" -Xmx8g \
  --enable-native-access=ALL-UNNAMED -cp target/classes \
  io.github.kirstenali.deepj.examples.MidTrainDeepJLoop100M

caffeinate -i "$JAVA_HOME/bin/java" -Xmx8g \
  --enable-native-access=ALL-UNNAMED -cp target/classes \
  io.github.kirstenali.deepj.examples.FineTuneDeepJLoop100M
```

Metal is selected when available. The entry points release backend resources
on completion. Two passes are used in every stage and saved as the inference
default.

## Resume and evaluation

Exported models continue to use model format 3. The architecture-aware
training-checkpoint version is a separate format, not the retired model format 2.

Rolling training checkpoints include weights, Adam state, data position,
schedule, architecture, pass count and residual scale. A mismatched configuration
is rejected before restoring parameters. Older parameter-only training checkpoints
cannot establish a looped model's configuration and are rejected for this model.
Other existing architectures retain support for their older training checkpoints.

```bash
caffeinate -i "$JAVA_HOME/bin/java" -Xmx8g \
  --enable-native-access=ALL-UNNAMED \
  -Ddeepj.resume=checkpoints/deepj-loop-100m/pretrain/training-latest.dj \
  -cp target/classes io.github.kirstenali.deepj.examples.TrainDeepJLoop100M

"$JAVA_HOME/bin/java" -Xmx8g --enable-native-access=ALL-UNNAMED \
  -cp target/classes io.github.kirstenali.deepj.examples.EvaluateDeepJLoop100M
```

Exported model files include the complete configuration and prompt format, so
`ModelLoader`, `DeepJChat`, and the desktop UI load them without manual settings.
Keep `tokenizer.bpe` alongside the model. Fine-tuned exports automatically use
the existing `Instruction:\n...\nResponse:\n` prompt format.

For a disposable pilot, override `deepj.steps`, `deepj.warmupSteps` and
`deepj.output`. A shortened pilot has a different learning-rate schedule and
is not a resumable starting point for the full schedule. Keep it separate.
Architecture overrides include `deepj.loops`, `deepj.preludeLayers`,
`deepj.coreLayers`, `deepj.codaLayers`, and `deepj.coreResidualScale`.
Use matching overrides when resuming a non-default configuration.

Compare held-out loss/perplexity and fixed-seed prompts, not only training loss.
An equal-token comparison is not an equal-compute comparison, because extra
passes require additional work.
