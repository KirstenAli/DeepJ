# DeepJ

DeepJ is a lightweight, **transformer-oriented** neural network library for Java.

## 📚 Examples

### 1) Classic ANN-style MLP (FNN)

```java

import org.deepj.ann.activations.GELU;
import org.deepj.ann.layers.FNN;
import org.deepj.ann.loss.MSELoss;
import org.deepj.ann.optimisers.AdamW;
import org.deepj.ann.training.SupervisedTraining;
import org.deepj.ann.training.Trainer;

import java.util.Random;

Random rnd = new Random(42);
FNN mlp = new FNN(
        3,
        new int[]{16, 16},
        3,
        GELU::new,   // activation factory (one instance per layer)
        null,
        rnd
);

Trainer trainer = SupervisedTraining.trainer(
        mlp,
        new MSELoss(),
        AdamW.defaultAdamW(3e-3),
        x,
        y,
        123L
);

trainer.

train(
    3000, // maxSteps
            3,    // batchSize
            200,  // logEvery
            0.98, // emaBeta
            1e-6  // targetEmaLoss
);
```

### 2) Transformer stack builder

Use `TransformerBuilder` to create a stack of decoder blocks.

```java
import org.deepj.ann.transformer.TransformerBuilder;
import org.deepj.ann.transformer.TransformerStack;
import org.deepj.ann.activations.GELU;

TransformerStack stack = new TransformerBuilder()
        .dModel(128)
        .nHeads(4)
        .dFF(512)
        .nLayers(4)
        .ffnActivation(GELU::new)
        .seed(42)
        .build();
```

### 3) Tiny GPT training + generation

```java
import org.deepj.ann.gpt.*;
import org.deepj.ann.tokenizer.ByteTokenizer;
import org.deepj.ann.training.CausalLMTraining;
import org.deepj.ann.training.Trainer;

import java.nio.file.Path;

Path corpus = Path.of("sample_data/sample_corpus.txt");

ByteTokenizer tok = new ByteTokenizer();
TextDataset ds = TextDataset.fromFile(corpus, tok, 64, 123);

GPTConfig cfg = new GPTConfig(
        ByteTokenizer.VOCAB_SIZE,
        64,     // maxSeqLen
        128,    // dModel
        4,      // nHeads
        2,      // nLayers
        4 * 128 // dFF
);

GPTModel model = new GPTModel(cfg, 42);

Trainer trainer = CausalLMTraining.trainer(model, ds, 1e-3);

// Train until target EMA loss or max steps.
trainer.

train(
        10_000, // maxSteps
                16,     // batchSize
                50,     // logEvery
                0.98,   // emaBeta
                0.25    // targetEmaLoss (tune based on corpus size)
);

// Generate a continuation.
String prompt = "Mara wrote down the rhythm, ";
String out = TextGenerator.generate(
        model,
        tok,
        cfg,
        prompt,
        64,    // maxNewTokens
        0.1,    // temperature
        0,     // topK (0 disables)
        1234L   // seed
);

System.out.

println("\n=== Generated ===");
System.out.

println(out);
```