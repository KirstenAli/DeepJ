# DeepJ 90M training

This is a new model. It does not replace the published TinyStories checkpoint.

## Architecture

| Setting | Value |
|---|---:|
| Parameters | 90,128,896 |
| Context length | 1,024 |
| Vocabulary | 16,384 |
| Transformer layers | 10 |
| Model width | 768 |
| Attention heads | 12 |
| Feed-forward width | 2,112 |
| Q rank | 384 |
| KV rank | 192 |

The smaller vocabulary leaves more parameters in the Transformer and reduces the largest training tensor. This is better suited to a 16 GB unified-memory Mac than copying the article's 65,536-token vocabulary.

## Data

The preparation script streams these datasets from Hugging Face:

| Stage | Dataset | Licence |
|---|---|---|
| Pretraining | FineWeb-Edu | ODC-By 1.0 |
| Conversation | Smol-SmolTalk | Apache 2.0 |
| Knowledge | MMLU | MIT |
| Mathematics | GSM8K | MIT |
| Science reasoning | AI2 ARC | CC BY-SA 4.0 |

The default FineWeb-Edu download is approximately 8.6 billion characters. The current tokenizer measured about 2.72 characters per token, so the downloaded corpus has more than the 1.8 billion tokens used by the default pretraining schedule. Pretraining reads the shuffled corpus sequentially and saves its exact position in each complete training checkpoint.

```shell
python3 -m venv .training-venv
source .training-venv/bin/activate
python3 -m pip install -r training-data-requirements.txt
python3 scripts/prepare_deepj_90m_data.py
```

Use smaller limits for a pipeline check:

```shell
python3 scripts/prepare_deepj_90m_data.py \
  --pretrain-characters 10000000 \
  --smoltalk-limit 1000 \
  --mmlu-limit 1000 \
  --gsm8k-limit 500 \
  --arc-limit 500
```

## Training

Build the project before starting:

```shell
mvn clean package
export DEEPJ_JDK=$(/usr/libexec/java_home -v 20)
```

Prepare the tokenizer independently:

```shell
"$DEEPJ_JDK/bin/java" \
  -Xmx4g \
  -Ddeepj.tokenizerOnly=true \
  --enable-native-access=ALL-UNNAMED \
  -cp target/classes \
  io.github.kirstenali.deepj.examples.TrainDeepJ90M
```

Pretrain on FineWeb-Edu:

```shell
caffeinate -i "$DEEPJ_JDK/bin/java" \
  -Xmx6g \
  --enable-native-access=ALL-UNNAMED \
  -cp target/classes \
  io.github.kirstenali.deepj.examples.TrainDeepJ90M
```

Continue with the conversation and reasoning mixture:

```shell
caffeinate -i "$DEEPJ_JDK/bin/java" \
  -Xmx6g \
  --enable-native-access=ALL-UNNAMED \
  -cp target/classes \
  io.github.kirstenali.deepj.examples.MidTrainDeepJ90M
```

Finish with response-only supervised fine-tuning:

```shell
caffeinate -i "$DEEPJ_JDK/bin/java" \
  -Xmx6g \
  --enable-native-access=ALL-UNNAMED \
  -cp target/classes \
  io.github.kirstenali.deepj.examples.FineTuneDeepJ90M
```

Each stage writes independent model and complete training checkpoints under `checkpoints/deepj-90m`. Resume a stage by passing its `training-latest.dj` file through `-Ddeepj.resume=...`.

Evaluate the final model:

```shell
"$DEEPJ_JDK/bin/java" \
  -Xmx6g \
  --enable-native-access=ALL-UNNAMED \
  -cp target/classes \
  io.github.kirstenali.deepj.examples.EvaluateDeepJ90M
```
