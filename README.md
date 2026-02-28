# DeepJ

DeepJ is a lightweight, **transformer-oriented** neural network library for Java.

## 📚 Examples

### 1) Classic ANN-style MLP (FNN)

Run: `org.DeepJ.examples.TrainClassicFNN`

```java
import org.DeepJ.ann.Tensor;
import org.DeepJ.ann.activations.GELU;
import org.DeepJ.ann.layers.FNN;
import org.DeepJ.ann.loss.MSELoss;
import org.DeepJ.ann.optimisers.AdamW;
import org.DeepJ.ann.training.SupervisedTraining;
import org.DeepJ.ann.training.Trainer;

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

trainer.train(
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
import org.DeepJ.ann.transformer.TransformerBuilder;
import org.DeepJ.ann.transformer.TransformerStack;
import org.DeepJ.ann.activations.GELU;

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

Run: `org.DeepJ.examples.TrainSmallGPT`

It trains a small GPT-style model on `sample_data/sample_corpus.txt` and then generates a continuation.
