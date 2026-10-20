# DeepJ Nova 100M

Nova is DeepJ's fourth decoder architecture. Its initial candidate has 24 layers, a model width of 576, a feed-forward width of 1,664, nine query heads, and three key/value heads. It uses RoPE, RMSNorm, bias-free SwiGLU projections, and shared input/output embedding weights. QK normalization is enabled by default and can be disabled with `-Ddeepj.qkNorm=false` before pretraining.

With 16,384 vocabulary entries and a 1,024-token context, the candidate has **99,711,552 unique trainable parameters**. Shared embeddings receive gradients from both uses and are updated and saved once. Checkpoints embed Nova's configuration and prompt format, so the desktop chat UI and `ModelLoader` can load them directly.

This is an untrained candidate. Its architecture is motivated by small-model research; improved answer quality and speed have not yet been demonstrated. Additional layers may increase activation memory and latency despite the similar parameter count. Training remains float32 and uses DeepJ's existing fused Metal operations.

## Data and stages

The examples use the same local files prepared for Frisky. They do not download, rewrite, or publish the corpora.

| Stage | Corpus | Default steps | Sampling |
|---|---|---:|---|
| Pretraining | `sample_data/deepj-90m/fineweb-edu.txt` | 1,757,813 | Sequential, excluding the same held-out tail |
| Mid-training | `sample_data/deepj-90m/midtrain-train.txt` | 256,000 | Random text windows |
| Fine-tuning | `sample_data/deepj-90m/sft-train.txt` | 22,500 | Response-only records |

Nova copies Frisky's existing `checkpoints/deepj-90m/pretrain/tokenizer.bpe` into its own output directory. It initializes new weights rather than loading Frisky's weights. Later stages load the preceding Nova stage's final checkpoint and preserve its embedded architecture. Instruction fine-tuning saves the same `Instruction:\n...\nResponse:\n` prompt format used by Frisky.

Checkpoints are saved under `checkpoints/deepj-100m/pretrain`, `midtrain`, and `sft`. Each stage remains a separate run so it can be evaluated before continuing. The examples automatically select Metal when available.

## Build and benchmark

Run commands from a checkout containing the local datasets and Frisky tokenizer, using JDK 20 or newer:

```sh
export JAVA_HOME=$(/usr/libexec/java_home -v 20)
mvn clean verify
```

Before committing to a full run, measure actual 1,024-token forward/backward/AdamW steps on the Mac:

```sh
caffeinate -i "$JAVA_HOME/bin/java" -Xmx8g --enable-native-access=ALL-UNNAMED -cp target/classes io.github.kirstenali.deepj.examples.BenchmarkDeepJ100M
```

The benchmark uses synthetic tokens, performs two warm-up steps and three measured steps, and saves nothing. Its loss is not a model-quality measurement. The full configuration's Metal throughput and memory requirements must be measured on the intended machine. If needed, `-Ddeepj.seqLen=64` runs a smaller memory probe first. Benchmarking requires Metal unless `-Ddeepj.allowCpuBenchmark=true` is explicitly set.

## Train

Start base pretraining:

```sh
caffeinate -i "$JAVA_HOME/bin/java" -Xmx8g --enable-native-access=ALL-UNNAMED -cp target/classes io.github.kirstenali.deepj.examples.TrainDeepJ100M
```

After evaluating the completed base checkpoint, run mid-training:

```sh
caffeinate -i "$JAVA_HOME/bin/java" -Xmx8g --enable-native-access=ALL-UNNAMED -cp target/classes io.github.kirstenali.deepj.examples.MidTrainDeepJ100M
```

After evaluating mid-training, run response-only fine-tuning:

```sh
caffeinate -i "$JAVA_HOME/bin/java" -Xmx8g --enable-native-access=ALL-UNNAMED -cp target/classes io.github.kirstenali.deepj.examples.FineTuneDeepJ100M
```

Each stage logs every 100 optimizer updates, saves resumable checkpoints every 1,000 updates, and releases temporary backend resources after every update. Batch size defaults to one 1,024-token sequence. Frisky's learning-rate defaults are retained as a baseline; they have not been tuned for Nova.

## Resume and evaluate

To resume, add `-Ddeepj.resume=checkpoints/deepj-100m/pretrain/training-latest.dj` before `-cp` in the pretraining command. Use the matching stage's path and main class for mid-training or fine-tuning. Resume restores the optimizer, schedule, dataset state, step count, and loss EMA. Keep the original architecture, dataset, batch size, and schedule settings when resuming.

Evaluate on the same held-out pretraining range and fixed validation seed used for Frisky:

```sh
"$JAVA_HOME/bin/java" -Xmx8g --enable-native-access=ALL-UNNAMED -cp target/classes io.github.kirstenali.deepj.examples.EvaluateDeepJ100M
```

Select a checkpoint with `-Ddeepj.checkpoint=...` and its tokenizer directory with `-Ddeepj.base=...`. The default validation tail is 64 MiB; retain the same `deepj.validationMiB` value used for training. Compare validation loss, factual and instruction checks, repetition across fixed seeds, and throughput before selecting a final configuration.

Paths can be overridden with `deepj.corpus`, `deepj.base`, `deepj.initialModel`, and `deepj.output`. Training overrides include `deepj.steps`, `deepj.batchSize`, `deepj.learningRate`, `deepj.minLearningRate`, and `deepj.warmupSteps`. An existing output containing checkpoints requires an explicit resume path or a different output directory.
