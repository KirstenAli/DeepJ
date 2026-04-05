<p align="center">
  <img src="./deepj_logo.svg" alt="DeepJ logo" width="170" />
</p>

# DeepJ

A lightweight, pure-Java GPT library — build, train, and experiment with Transformer models, zero dependencies.

📚 **Javadoc:** https://kirstenali.github.io/DeepJ/

---

## 🚀 Installation

Add the GitHub Packages repository and dependency to your `pom.xml`.

```xml
<repositories>
    <repository>
        <id>github</id>
        <url>https://maven.pkg.github.com/KirstenAli/DeepJ</url>
    </repository>
</repositories>

<dependencies>
    <dependency>
        <groupId>io.github.kirstenali</groupId>
        <artifactId>deepj</artifactId>
        <version>0.4.5-alpha</version>
    </dependency>
</dependencies>
```

---

## 🏗️ Architecture

DeepJ is organised into focused packages:

| Package | Purpose |
|---|---|
| `tensor` | 2-D tensors, `TensorBackend` abstraction, `ComputeGraph` lazy execution |
| `tensor.cpu` | `CpuBackend` — parallel CPU implementation with in-place helpers |
| `tensor.metal` | `MetalBackend` — Apple Silicon GPU via Metal JNI |
| `concurrent` | `DeepJExecutor` — thread-pool for CPU parallelism |
| `layers` | `Layer` interface, `Linear` projection, `FNN` (MLP) |
| `layers.transformer` | `TransformerBlock`, multi-head attention, layer norm |
| `transformer` | `TransformerBuilder`, `TransformerStack` |
| `transformer.embeddings` | `Embedding` (token lookup), `PositionalEmbedding` (learnable) |
| `activations` | `ActivationFunction` interface + GELU, ReLU, Sigmoid, Tanh, Softmax |
| `loss` | `LossFunction` interface + MSELoss, CrossEntropyLoss |
| `optimisers` | `ParameterOptimizer` interface, `AdamW`, `Parameter` |
| `training` | `Trainer` loop, `Trainable`, `SupervisedTraining`, `CausalLMTraining` |
| `models.gpt` | `GPTModel`, `GPTConfig`, `TextGenerator` |
| `tokenizers` | `Tokenizer` interface, `ByteTokenizer`, BPE pipeline |
| `data` | `TextDataset`, `Batch` |
| `persistence` | `Persistable` interface, `ModelSerializer` (binary save/load) |
| `chatui` | `BaseChatApp`, `ChatService` — optional JavaFX chat interface |

### Key interfaces

```
Trainable              — exposes parameters() + zeroGrad()
    └── Layer          — forward(Tensor) / backward(Tensor)
Persistable            — save(Path) / load(Path) via ModelSerializer
TensorBackend          — all tensor ops, swappable (CpuBackend / MetalBackend)
LossFunction           — loss() + gradient()
ParameterOptimizer     — step(List<Parameter>)
Tokenizer              — encode(String) / decode(int[]) / vocabSize()
```

---

## 📐 Design Decisions

### Backend abstraction

Every tensor operation routes through a pluggable `TensorBackend`.
The default is `CpuBackend`; swap to `MetalBackend` for GPU acceleration:

```java
Tensor.setBackend(new MetalBackend());  // GPU-backed where kernels exist; otherwise CPU fallback
Tensor.setBackend(new CpuBackend());    // back to CPU
```

No other code changes — models, training loops, and layers are
backend-agnostic.

### Lazy GPU execution

When the Metal backend is active, ops are **recorded**, not executed
immediately. A flat `int[]` command stream accumulates op codes and
buffer IDs. Everything flushes in a single native call:

```
record matmul  →  record relu  →  record softmax  →  flush (one GPU dispatch)
```

Tensors stay GPU-resident between ops — data only transfers on explicit
materialization/accessor calls (`materialize`, `get`, `set`, etc.) or
when a CPU-fallback op needs it.

### Zero-allocation in-place ops (CPU)

On the CPU backend, every element-wise op has an in-place variant that
writes the result back into the input tensor — **no `new Tensor()`
allocation**:

```java
// Allocating (returns a new tensor):
Tensor result = a.sqrt();

// In-place (mutates a, zero allocation):
a.sqrtInPlace();

// Chaining:
gradient.multiplyScalarInPlace(0.5).addInPlace(bias).reluInPlace();
```

`CpuBackend` overrides every in-place method for true zero-allocation —
the function writes directly into the input's `data[][]`, no temporary
tensor is created. Other backends (e.g. `MetalBackend`) inherit working
default implementations that allocate a temporary internally.

### CPU parallelism

`DeepJExecutor.forRange()` splits work across a daemon thread pool.
Enabled by default; toggle with a flag:

```java
DeepJExecutor.setParallelEnabled(false);  // single-threaded
DeepJExecutor.setParallelEnabled(true);   // multi-threaded (default)
DeepJExecutor.setNumThreads(8);           // override thread count
```

---

## 📊 Tensor

`Tensor` is a 2-D matrix of `double` values — the core data type for
every operation in DeepJ. All methods route through the active
`TensorBackend`, so the same code runs on CPU or GPU.

### Creating tensors

```java
// Zero-filled
Tensor a = new Tensor(3, 4);              // 3 rows × 4 cols, all zeros

// From data
Tensor b = new Tensor(new double[][]{
        {1, 2, 3},
        {4, 5, 6}
});                                        // 2 × 3, deep-copied

// Static factories
Tensor z = Tensor.zeros(4, 4);
Tensor o = Tensor.ones(4, 4);
Tensor r = Tensor.random(4, 4, new Random(42));  // Gaussian × 0.1
Tensor m = Tensor.causalMask(8);                  // 8 × 8, upper triangle = -1e9
```

### Shape

```java
a.rows   // number of rows
a.cols   // number of columns
a.data   // raw double[][] (trigger materialize() first if on GPU)
```

### Operations by category

#### Element-wise binary (same shape required)

```java
a.add(b)        a.subtract(b)        a.multiply(b)        a.divide(b)
a.matmul(b)     // [m×k] · [k×n] → [m×n]
```

#### Scalar

```java
a.multiplyScalar(2.0)     a.addScalar(1.0)     a.divideScalar(3.0)
```

#### Broadcast

```java
// Row vector [1×cols] broadcast across all rows
a.addRowVector(rv)        a.addBroadcastRows(rv)     a.multiplyBroadcastRows(rv)

// Column vector [rows×1] broadcast across all cols
a.addBroadcastCols(cv)    a.subtractBroadcastCols(cv)
a.multiplyBroadcastCols(cv)    a.divideBroadcastCols(cv)
```

#### Unary math

```java
a.sqrt()    a.neg()     a.exp()     a.log()
a.pow(2.0)  a.clamp(0, 1)           a.transpose()
```

#### Activations

```java
a.reluActivation()    a.geluActivation()    a.tanhActivation()    a.sigmoidActivation()
a.reluBackward(grad)  a.geluBackward(grad)
```

#### Reductions

```java
a.sumRows()          // [1 × cols] — sum each column across rows
a.sumAlongRows()     // [rows × 1] — sum each row
a.sumAlongCols()     // [1 × cols]
a.meanAlongRows()    // [rows × 1]
a.varianceAlongRows() // [rows × 1]
a.maxAlongRows()     // [rows × 1]
a.sum()              // scalar — total sum
a.sumAbs()           // scalar — total absolute sum
```

#### Row-wise compound

```java
a.softmaxRows()                  // row-wise softmax
a.softmaxBackward(softmaxOut)    // softmax backward
a.crossEntropyLoss(targets)      // scalar loss
a.crossEntropyGradient(targets)  // gradient tensor
```

#### Data access (triggers materialization on GPU)

```java
a.get(r, c)                     // read one element
a.set(r, c, value)              // write one element
a.getRow(r)                     // [1 × cols] copy
Tensor.sliceRows(t, indices, cols)   // gather rows by index
Tensor.scatterAddRows(t, indices, g) // scatter-add (in-place)
Tensor.sampleRows(t, n, rnd)        // random row sample
```

### In-place ops (zero allocation on CPU)

Every element-wise op has an `*InPlace` variant that mutates the tensor
and returns `this` for chaining:

```java
a.sqrtInPlace();
a.multiplyScalarInPlace(0.5).addInPlace(b).reluInPlace();
```

Full list: `addInPlace`, `subtractInPlace`, `multiplyInPlace`,
`divideInPlace`, `multiplyScalarInPlace`, `addScalarInPlace`,
`divideScalarInPlace`, `sqrtInPlace`, `negInPlace`, `expInPlace`,
`logInPlace`, `reluInPlace`, `geluInPlace`, `tanhInPlace`,
`sigmoidInPlace`.

### GPU materialization

When the Metal backend is active, `data[][]` may be stale. Call
`materialize()` before reading raw data, or use accessor methods
(`get`, `set`, `getRow`, `sum`, `print`) which materialise automatically:

```java
a.materialize();         // flush GPU ops, download result
double v = a.get(0, 0); // auto-materialises
a.print("my tensor");   // auto-materialises
```

### Flatten / unflatten

```java
double[] flat = Tensor.flattenTensor(a);                 // row-major flat array
Tensor t = Tensor.unflattenToTensor(flat, rows, cols);   // back to 2-D
```

---

## ⚙️ Concurrency (`DeepJExecutor`)

All CPU-side tensor parallelism flows through a single class:
`DeepJExecutor`. It manages a daemon thread pool and exposes one
primitive — `forRange(start, end, body)` — used by every `CpuBackend`
operation.

### How it works

```
forRange(0, rows, r -> { process row r })
        │
        ├── parallel disabled or 1 thread?  → run sequentially on caller thread
        │
        └── otherwise → split [0, rows) into chunks, submit to thread pool
                         wait for all chunks via CountDownLatch
                         fail-fast: if any chunk throws, cancel the rest
```

### Configuration

```java
import io.github.kirstenali.deepj.concurrent.DeepJExecutor;

// Toggle parallelism
DeepJExecutor.setParallelEnabled(true);   // default
DeepJExecutor.setParallelEnabled(false);  // useful for debugging

// Check state
boolean on = DeepJExecutor.isParallelEnabled();

// Override thread count (default: availableProcessors - 1)
DeepJExecutor.setNumThreads(4);
int n = DeepJExecutor.getNumThreads();

// Clean shutdown (optional — daemon threads exit with the JVM)
DeepJExecutor.shutdown();
```

### Design notes

-   **Daemon threads** — the pool never prevents JVM exit
-   **Core timeout** — idle threads expire after 30 seconds
-   **Fail-fast** — an `AtomicBoolean` flag cancels remaining chunks
    if any chunk throws a `RuntimeException`
-   **O(1) memory** — chunk boundaries are computed on the fly, not stored

---

## 📚 Examples

### Classic MLP (FNN)

Train a small feed-forward neural network with supervised training.

```java
Tensor x = new Tensor(new double[][]{
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
});

Tensor y = new Tensor(new double[][]{
        {0, 0, 1},
        {0, 1, 0},
        {1, 0, 0}
});

Random rnd = new Random(42);

FNN mlp = new FNN(
        3,                 // inputSize
        new int[]{16, 16}, // hiddenSizes
        3,                 // outputSize
        GELU::new,         // hiddenActivation
        null,              // outputActivation (none)
        rnd                // random seed source
);

Trainer trainer = SupervisedTraining.trainer(
        mlp,
        new MSELoss(),
        AdamW.defaultAdamW(3e-3), // lr
        x,
        y,
        123L                      // seed
);

trainer.train(
        3000, // maxSteps         – stop after 3 000 gradient updates
        3,    // batchSize        – rows sampled per step
        200,  // logEvery         – print loss every 200 steps
        0.98, // emaBeta          – smoothing factor for the moving-average loss
        1e-6  // targetEmaLoss    – early-stop when smoothed loss falls below this
);
```

### Transformer stack

Create a stack of decoder blocks using the builder.

```java
TransformerStack stack = new TransformerBuilder()
        .dModel(128)          // model dimension
        .nHeads(4)            // attention heads
        .dFF(512)             // feed-forward inner dimension
        .nLayers(4)           // number of decoder blocks
        .ffnActivation(GELU::new) // activation inside FFN (default: GELU)
        .seed(42)             // weight initialisation seed
        .build();
```

Each block contains multi-head self-attention, layer norm, and an FFN.
The stack implements `Layer`, so you can call `forward()` / `backward()`
and collect `parameters()` like any other layer.

### GPT training + generation

Train a small GPT-style language model.

```java
// ── Backend ───────────────────────────────────────────────────────────
MetalBackend metal = new MetalBackend();
Tensor.setBackend(metal);

// ── Data ──────────────────────────────────────────────────────────────
Path corpus = Path.of("sample_data/llm_training_dataset_1227_examples.txt");

Tokenizer tok = new ByteTokenizer();
TextDataset ds = TextDataset.fromFile(corpus, tok, 256, 123);

// ── Model ─────────────────────────────────────────────────────────────
GPTConfig cfg = new GPTConfig(
        tok.vocabSize(),
        256,  // maxSeqLen
        512,  // dModel
        4,    // nHeads
        5,    // nLayers
        1025  // dFF
);

GPTModel model = new GPTModel(cfg, 42);

// ── Training ──────────────────────────────────────────────────────────
Trainer trainer = CausalLMTraining.trainer(model, ds, 1e-4);

// ── Checkpointing ─────────────────────────────────────────────────────
Path checkpointDir = Path.of("checkpoints");

Trainer.StepHook checkpointHook = (step, loss, ema) -> {
    if (step > 0 && step % 500 == 0) {
        model.save(checkpointDir.resolve("small-gpt-" + step + ".bin"));
    }
};

trainer.train(
        10_000_000,    // maxSteps            – stop after 10 M gradient updates
        2,             // batchSize           – sequences per step
        1,             // logEvery            – print loss every step
        0.98,          // emaBeta             – smoothing factor for the moving-average loss
        0.01,          // targetEmaLoss       – early-stop when smoothed loss falls below this
        25,            // releaseEverySteps   – free orphaned GPU buffers every 25 steps
        checkpointHook // called after each step
);

Path finalModelPath = checkpointDir.resolve("small-gpt-final.bin");
model.save(finalModelPath);

// ── Inference ─────────────────────────────────────────────────────────
GPTModel loadedModel = new GPTModel(cfg, 42);
loadedModel.load(finalModelPath);

String prompt = "Bob Marley was ";

String out = TextGenerator.generate(
        loadedModel,   // model
        tok,           // tokenizer
        cfg,           // config
        prompt,        // prompt text
        200,           // maxNewTokens   – generate up to 200 tokens after the prompt
        0.1,           // temperature    – low = more deterministic, high = more random
        20,            // topK           – sample from the 20 most likely tokens (0 = full vocab)
        1234L          // seed           – for reproducible sampling
);

System.out.println("\n=== Generated ===");
System.out.println(out);
```

---

## 🧱 Layers

All layers implement the `Layer` interface — `forward(Tensor)`,
`backward(Tensor)`, and `parameters()` for optimizer access.

### Linear

Fully-connected projection: `y = xW + b`.

```java
Linear fc = new Linear(
        128,  // inputSize
        64,   // outputSize
        rnd   // Random for weight init
);

Tensor y = fc.forward(x);       // [n x 64]
Tensor dX = fc.backward(gradY); // backprop, accumulates W.grad and b.grad
```

### FNN (MLP)

Stack of `Linear` → activation → `Linear` → activation → … → `Linear`.

```java
FNN mlp = new FNN(
        784,               // inputSize
        new int[]{256, 128}, // hiddenSizes
        10,                // outputSize
        GELU::new,         // hiddenActivation (factory — each layer gets its own)
        null,              // outputActivation (none)
        rnd
);

Tensor out = mlp.forward(x);
Tensor dX = mlp.backward(gradOut);
List<Parameter> params = mlp.parameters(); // all W and b from every Linear
```

### Transformer layers

| Class | Description |
|---|---|
| `LayerNorm1D` | Layer norm over feature dimension with trainable γ/β |
| `MultiHeadSelfAttention` | Multi-head causal self-attention (`[seqLen × dModel]`) |
| `TransformerBlock` | Pre-LN decoder block: `x + Attn(LN(x))`, then `x + FFN(LN(x))` |

Use individually or via `TransformerBuilder`:

```java
// Individual block
TransformerBlock block = new TransformerBlock(
        512,       // dModel
        8,         // nHeads
        2048,      // dFF
        GELU::new, // FFN activation
        rnd
);

Tensor out = block.forward(x);   // [seqLen x dModel]
Tensor dX = block.backward(grad);

// Or stack via builder
TransformerStack stack = new TransformerBuilder()
        .dModel(512).nHeads(8).dFF(2048).nLayers(6)
        .ffnActivation(GELU::new).seed(42)
        .build();
```

### Custom layers

Implement `Layer` to create your own:

```java
public class MyLayer implements Layer {
    @Override public Tensor forward(Tensor input)      { /* ... */ }
    @Override public Tensor backward(Tensor gradOutput) { /* ... */ }
    @Override public List<Parameter> parameters()       { return List.of(); }
}
```

Any `Layer` works with `SupervisedTraining`, `AdamW`, and `ModelSerializer`
out of the box.

---

## 🔧 Transformer (`TransformerBuilder` / `TransformerStack`)

`TransformerBuilder` assembles a `TransformerStack` — a sequential
chain of `TransformerBlock`s that implements `Layer`.

### Builder methods

| Method | Default | Description |
|---|---|---|
| `.dModel(int)` | required | Model / embedding dimension |
| `.nHeads(int)` | required | Number of attention heads (`dModel` must be divisible) |
| `.dFF(int)` | required | Feed-forward inner dimension |
| `.nLayers(int)` | required | Number of decoder blocks |
| `.ffnActivation(Supplier)` | `GELU::new` | Activation inside each FFN |
| `.seed(long)` | `42` | Weight initialisation seed |
| `.random(Random)` | — | Provide your own `Random` (overrides seed) |

### TransformerStack

`TransformerStack` is a `record` that implements `Layer`:

```java
TransformerStack stack = new TransformerBuilder()
        .dModel(256).nHeads(4).dFF(1024).nLayers(6)
        .ffnActivation(GELU::new).seed(42)
        .build();

Tensor out = stack.forward(x);          // [seqLen x dModel]
Tensor dX  = stack.backward(gradOut);   // backprop through all blocks in reverse
List<Parameter> ps = stack.parameters(); // all block parameters
```

The stack is used internally by `GPTModel` and can also be used
standalone for custom architectures.

---

## 🤖 Models

### GPTModel

Decoder-only GPT-style transformer. Composes token embeddings,
positional embeddings, a `TransformerStack`, final layer norm, and a
linear head.

```java
GPTConfig cfg = new GPTConfig(
        vocabSize,    // number of tokens
        maxSeqLen,    // context window length
        dModel,       // model dimension
        nHeads,       // attention heads
        nLayers,      // decoder blocks
        dFF           // FFN inner dimension
);

GPTModel model = new GPTModel(cfg, 42); // seed
```

#### GPTConfig

| Field | Default | Description |
|---|---|---|
| `vocabSize` | required | Token vocabulary size |
| `maxSeqLen` | required | Maximum sequence length |
| `dModel` | required | Embedding / model dimension |
| `nHeads` | required | Attention heads (`dModel % nHeads == 0`) |
| `nLayers` | required | Number of transformer blocks |
| `dFF` | required | Feed-forward inner dimension |
| `initScale` | `0.2` | Multiply all initial weights by this factor |
| `gradClipNorm` | `1.0` | Global gradient clipping threshold |

The 6-arg constructor uses defaults for `initScale` and `gradClipNorm`.
The full 8-arg constructor allows overriding both.

#### Forward / backward

```java
Tensor logits = model.forward(inputIds);   // int[] → [seqLen x vocabSize]
model.backward(dLogits);                    // backprop through entire model
List<Parameter> params = model.parameters(); // all trainable weights
```

#### Architecture

```
inputIds
  → token embedding + positional embedding
  → TransformerStack (nLayers × TransformerBlock)
  → LayerNorm
  → Linear head → logits [seqLen × vocabSize]
```

### TextGenerator

Autoregressive sampling from a trained `GPTModel`:

```java
String out = TextGenerator.generate(
        model,    // trained GPTModel
        tok,      // Tokenizer
        cfg,      // GPTConfig
        "Hello",  // prompt
        200,      // maxNewTokens   – generate up to 200 tokens after the prompt
        0.8,      // temperature    – low = more deterministic, high = more random
        20,       // topK           – sample from the 20 most likely tokens (0 = full vocab)
        42L       // seed           – for reproducible sampling
);
```

Internally it runs an autoregressive loop: encode prompt → forward →
sample from top-k logits with temperature → append token → repeat.

---

## 🔤 Tokenizers

DeepJ provides a `Tokenizer` interface with two implementations:

### ByteTokenizer

256-token byte-level tokenizer. No training needed — every byte maps
to a token. Good for quick experiments.

```java
Tokenizer tok = new ByteTokenizer();   // vocabSize = 256
int[] ids = tok.encode("hello");
String text = tok.decode(ids);
```

### BPE tokenizer

Full byte-pair encoding pipeline: train a merge table from text, then
encode/decode with subword tokens.

```java
// Train a BPE model from a corpus
BPEModel bpe = new BPETrainer().train(corpusText, 1000);  // target vocab size

// Or from a file
BPEModel bpe = new BPETrainer().trainFromFile(Path.of("corpus.txt"), 1000);

// Wrap as a Tokenizer
Tokenizer tok = new BPETokenizer(bpe);
int[] ids = tok.encode("hello world");
String text = tok.decode(ids);
```

---

## 📂 Data Loading

### TextDataset

Simple in-memory dataset that tokenises a text file and samples random
contiguous chunks for causal language model training.

```java
// From a file
TextDataset ds = TextDataset.fromFile(
        Path.of("corpus.txt"),
        tok,    // any Tokenizer
        256,    // seqLen – context window length
        123     // seed
);

// Or from pre-tokenised ids
TextDataset ds = new TextDataset(tokenIds, 256, 123);

// Sample a batch
Batch batch = ds.nextBatch(4);  // batchSize = 4
// batch.x() → int[4][256] (input ids)
// batch.y() → int[4][256] (target ids, shifted by 1)
```

Each batch contains `x` (input tokens) and `y` (next-token targets),
both shaped `[batchSize][seqLen]`.

---

## 🏋️ Training

### Trainer

`Trainer` is a reusable training loop that delegates each step to a
pluggable `StepFunction`. It handles EMA loss tracking, logging,
periodic GPU resource release, and early stopping.

#### StepFunction

The core contract — one optimisation step, returns the loss:

```java
@FunctionalInterface
public interface StepFunction {
    double trainStep(int batchSize);
}
```

You can write your own or use the factories below.

#### `train()` overloads

```java
// Minimal (uses default releaseEverySteps=25, no hook)
trainer.train(maxSteps, batchSize, logEvery, emaBeta, targetEmaLoss);

// With custom GPU release cadence
trainer.train(maxSteps, batchSize, logEvery, emaBeta, targetEmaLoss, releaseEverySteps);

// With step hook (default release cadence)
trainer.train(maxSteps, batchSize, logEvery, emaBeta, targetEmaLoss, stepHook);

// Full control
trainer.train(maxSteps, batchSize, logEvery, emaBeta, targetEmaLoss, releaseEverySteps, stepHook);
```

#### What the loop does

```
for each step:
    1. call stepFn.trainStep(batchSize)  → loss
    2. update EMA loss
    3. log if step % logEvery == 0
    4. call stepHook (if provided)
    5. releaseResources if step > 0 && step % releaseEverySteps == 0
    6. early-stop if ema <= targetEmaLoss
finally:
    releaseResources (always, even on exception)

When using `MetalBackend`, `releaseResources()` materializes tracked stale
tensors before buffers are cleared so optimizer/model state is preserved.
```

### SupervisedTraining

Factory for Tensor-in / Tensor-out models (FNN, etc.). Each step:

1.  `zeroGrad()`
2.  Sample a mini-batch from `(x, y)` (or use all rows if `batchSize ≥ rows`)
3.  `forward()` → `loss()` → `gradient()` → `backward()`
4.  `opt.step(parameters())`

```java
Trainer trainer = SupervisedTraining.trainer(
        model,                    // any Layer (FNN, custom, etc.)
        new MSELoss(),            // or CrossEntropyLoss
        AdamW.defaultAdamW(1e-3), // optimizer
        x, y,                     // training data
        42L                       // seed
);
```

### CausalLMTraining

Factory for autoregressive language models (GPT). Each step:

1.  `zeroGrad()`
2.  Sample a `Batch` from `TextDataset`
3.  For each sequence in the batch: `forward()` → `crossEntropyLoss()` → `backward()`
4.  Average gradients across the batch
5.  Global gradient clipping (using `cfg.gradClipNorm()`)
6.  `opt.step(parameters())`

```java
Trainer trainer = CausalLMTraining.trainer(model, dataset, 1e-4);
```

### StepHook

Hook into every training step for checkpointing, custom logging, etc.:

```java
Trainer.StepHook hook = (step, loss, ema) -> {
    if (step % 500 == 0) model.save(Path.of("ckpt-" + step + ".bin"));
};
```

### TrainingResult

`trainer.train()` returns a `TrainingResult` record:

```java
TrainingResult result = trainer.train(...);
result.steps();    // total steps completed
result.lastLoss(); // loss on the final step
result.emaLoss();  // exponential moving average loss at finish
```

---

## 🧮 Optimizers

### AdamW

AdamW with per-parameter state, bias correction, and weight decay.

```java
// With defaults (beta1=0.9, beta2=0.999, eps=1e-8, weightDecay=0.01)
AdamW opt = AdamW.defaultAdamW(1e-3);

// Full control
AdamW opt = new AdamW(1e-3, 0.9, 0.999, 1e-8, 0.01);

// Step
opt.step(model.parameters());
```

---

## 📉 Loss Functions

| Class | Formula | Use case |
|---|---|---|
| `MSELoss` | mean((pred − actual)²) | Regression, autoencoders |
| `CrossEntropyLoss` | −log(softmax(pred)[target]) | Classification |

```java
LossFunction loss = new MSELoss();
double l = loss.loss(predicted, actual);
Tensor grad = loss.gradient(predicted, actual);
```

---

## 🔥 Activations

DeepJ provides an `ActivationFunction` interface with `forward()` /
`backward()`. Five implementations are included:

| Class | Function |
|---|---|
| `GELU` | Gaussian Error Linear Unit |
| `ReLU` | max(0, x) |
| `Sigmoid` | 1 / (1 + e⁻ˣ) |
| `Tanh` | hyperbolic tangent |
| `Softmax` | row-wise softmax |

Activations are passed as factories (`Supplier<ActivationFunction>`)
so each layer gets its own instance — no shared state during backprop:

```java
FNN mlp = new FNN(inputSize, hiddenSizes, outputSize, GELU::new, null, rnd);

TransformerStack stack = new TransformerBuilder()
        .ffnActivation(ReLU::new)  // swap activation
        // ...
        .build();
```

---

## 💾 Persistence

Any model implementing `Persistable` (e.g. `GPTModel`) gets
binary save/load for free via `ModelSerializer`:

```java
// Save all parameters to a binary file
model.save(Path.of("model.bin"));

// Load parameters back (model structure must match)
model.load(Path.of("model.bin"));
```

The serializer writes raw `double` values in order — compact and fast,
no external format dependencies.

---

## 🖥️ Metal GPU Backend (macOS)

DeepJ ships with an optional **Metal GPU backend** for macOS Apple Silicon.
GPU kernels cover core training-path ops (matmul, element-wise math,
softmax/softmax backward, layernorm backward, AdamW update, and several
broadcast/reduction ops). Ops without Metal kernels still fall back to CPU.
CPU ↔ GPU transfer is automatic and only happens when materialization is
required.

### Enabling the backend

```java
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;

MetalBackend metal = new MetalBackend();
Tensor.setBackend(metal);
```

That's it — all `Tensor` operations route through the same backend entrypoint.
If a Metal kernel exists, the op is recorded for GPU execution; otherwise it
falls back to CPU with on-demand materialization.

### GPU-resident tensor design

DeepJ uses a **lazy, GPU-resident** execution model. When the Metal backend
is active, tensors stay on the GPU between operations — data is only
transferred when absolutely necessary:

```
CPU                              GPU
 │                                │
 │  Tensor.setBackend(metal)      │
 │                                │
 │  a = random(512, 512)     ──upload──▶  GPU buffer A
 │  b = random(512, 512)     ──upload──▶  GPU buffer B
 │                                │
 │  c = a.matmul(b)               │       record op  (no execution yet)
 │  d = c.relu()                  │       record op  (no execution yet)
 │  e = d.softmaxRows()           │       record op  (no execution yet)
 │                                │
 │  e.materialize()               │       flush all 3 ops in one batch
 │                           ◀──download── GPU buffer E
 │  read e.data[][]               │
```

Key points:

-   **No round-trips (GPU-supported chains):** intermediate results (`c`, `d`) stay on GPU
-   **Batched execution:** all recorded ops flush in a single GPU command buffer
-   **Materialization on demand:** call `materialize()` (or use accessors like `get`, `set`, `print`) before reading CPU data
-   **Staleness tracking:** each tensor knows whether its CPU or GPU copy is current via `GpuBuffer.cpuStale` / `needsUpload` flags
-   **Zero-copy reuse:** a tensor's GPU buffer persists across operations — subsequent ops reuse it without re-uploading
-   **CPU-fallback ops** materialise tensors on demand and delegate to `CpuBackend`

A full forward + backward pass records hundreds of GPU ops and executes them
in just 2–3 native calls, minimising driver overhead.

---

## ⚡ Metal GPU Performance

Chained-pipeline benchmarks on a **512 × 512** matrix (262,144 elements),
measured on Apple Silicon with `-Dperf.iters.cpu=3` and `-Dperf.iters.gpu=10`.
Multiple lazy GPU ops are recorded into one command buffer, then a single
`materialize()` flushes them all — matching how the GPU runs during training.
CPU timings use the default multithreaded `DeepJExecutor` (parallel enabled).

| Pipeline | CPU (ms) | GPU (ms) | Speedup |
|---|---:|---:|---:|
| 5 mixed ops (1 matmul) | 19.296 | 1.686 | 11.45× |
| 10 mixed ops (2 matmuls) | 40.876 | 1.750 | 23.35× |
| 20 mixed ops (4 matmuls) | 104.231 | 3.147 | 33.12× |
| 50 mixed ops (10 matmuls) | 226.345 | 6.902 | 32.80× |
| linear fwd (3 ops) | 18.967 | 1.358 | 13.96× |
| attention fwd (4 ops) | 40.332 | 1.594 | 25.30× |
| fwd + loss grad (6 ops) | 40.645 | 2.013 | 20.19× |
| backward (6 ops, 2 matmuls) | 42.140 | 1.736 | 24.28× |
| mini train step (9 ops) | 46.088 | 2.014 | 22.89× |
| full train step (13 ops) | 85.293 | 2.383 | 35.79× |

> **Key takeaway:** the lazy compute graph batches many GPU kernels into a
> single command buffer. As chain depth grows the speedup climbs — a full
> forward + backward + optimise step runs **~36× faster** on Metal vs
> multithreaded CPU.

Run the benchmark yourself:

```bash
mvn test -Dtest=MetalBackendAllOpsPerformanceTest \
    "-Djunit.jupiter.conditions.deactivate=org.junit.*" \
    -Dperf.size=512 -Dperf.iters.cpu=3 -Dperf.iters.gpu=10 \
    -Dsurefire.useFile=false
```

---

## 💬 Chat UI

DeepJ includes an optional **JavaFX chat interface** for interacting
with trained models. The UI is model-agnostic — you provide your own
`ChatService` implementation.

### Launching

Extend `BaseChatApp` and provide your service:

```java
public class ChatApp extends BaseChatApp {

    @Override
    protected ChatService createChatService() {
        return new GPTChatService();
    }

    public static void main(String[] args) {
        launch();
    }
}
```

### Implementing ChatService

Your service controls model loading, tokenization, and generation:

```java
public class GPTChatService implements ChatService {

    private GPTModel model;
    private final Tokenizer tokenizer = new ByteTokenizer();

    private final GPTConfig config = new GPTConfig(
            ByteTokenizer.VOCAB_SIZE,
            128,  // maxSeqLen
            256,  // dModel
            4,    // nHeads
            4,    // nLayers
            1024  // dFF
    );

    @Override
    public void loadModel(Path modelPath) throws Exception {
        model = new GPTModel(config, 42);
        model.load(modelPath);
    }

    @Override
    public boolean isModelLoaded() {
        return model != null;
    }

    @Override
    public String getLoadedModelName() {
        return "my-gpt";
    }

    @Override
    public String generate(String prompt, int maxTokens, double temperature, int topK, long seed) {
        return TextGenerator.generate(
                model, tokenizer, config, prompt,
                maxTokens, temperature, topK, seed
        );
    }
}
```

### UI flow

1.  User selects a trained `.bin` model
2.  `ChatService.loadModel()` loads the model
3.  User enters a prompt
4.  The UI calls `chatService.generate(...)`
5.  Generated text appears in the chat window

---

## 📄 License

MIT License.
