<p align="center">
  <img src="./deepj_logo.svg" alt="DeepJ logo" width="170" />
</p>

# DeepJ

A lightweight, pure-Java Transformer library — build, train, and experiment with GPT, Llama, and DeepSeek-style models, zero dependencies.

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
        <version>0.6.0-alpha</version>
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
| `layers` | `Layer` interface, `Projection` interface (sub-interface for linear projections), `Linear`, `FNN` (MLP) |
| `layers.transformer` | `SwiGLULayer` — gated FFN used by Llama / DeepSeek style blocks |
| `layers.transformer.attention` | `MultiHeadSelfAttention`, `RoPEMultiHeadSelfAttention`, `MultiHeadLatentAttention` (MLA) |
| `layers.transformer.norm` | `NormLayer` interface, `LayerNorm1D`, `RMSNorm1D` |
| `layers.transformer.blocks` | `GPTTransformerBlock`, `LlamaTransformerBlock`, `DeepSeekTransformerBlock` |
| `transformer` | `GPTTransformerBuilder`, `LlamaTransformerBuilder`, `DeepSeekTransformerBuilder`, `TransformerStack` |
| `transformer.embeddings` | `Embedding` (token lookup), `PositionalEmbedding` (learnable), `RotaryEmbedding` (RoPE) |
| `activations` | `ActivationFunction` interface + GELU, SiLU, ReLU, Sigmoid, Tanh, Softmax |
| `loss` | `LossFunction` interface + MSELoss, CrossEntropyLoss |
| `optimisers` | `ParameterOptimizer` interface, `AdamW`, `Parameter` |
| `training` | `Trainer` loop, `Trainable`, `SupervisedTraining`, `CausalLMTraining` |
| `models` | `TransformerConfig` (shared config interface), `CausalLM` (shared model interface), `DecoderOnlyModel` (abstract base), `TextGenerator` |
| `models.gpt` | `GPTModel`, `GPTConfig` |
| `models.llama` | `LlamaModel`, `LlamaConfig` |
| `models.deepseek` | `DeepSeekModel`, `DeepSeekConfig` |
| `tokenizers` | `Tokenizer` interface, `ByteTokenizer`, BPE pipeline (`BPETrainer`, `BPETokenizer`, `BPEModelIO`) |
| `data` | `TextDataset` (streaming, memory-mapped), `Batch` |
| `persistence` | `Persistable` interface, `ModelSerializer` (binary save/load) |
| `chatui` | `BaseChatApp`, `ChatService` — optional JavaFX chat interface |


---

## 📐 Design Decisions

### Backend abstraction

Every tensor operation routes through a pluggable `TensorBackend`.
The default is `CpuBackend`; swap to `MetalBackend` for GPU acceleration:

```java
Tensor.setBackend(new MetalBackend());  // route tensor ops through lazy Metal compute graph
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
when CPU-side reads/writes are requested.

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
the function writes directly into the input's flat `data[]`, no temporary
tensor is created. `MetalBackend` also overrides in-place ops to keep
execution lazy and GPU-resident (no forced CPU materialization in the
hot training path). In-place GPU results are rebound through
`ComputeGraph` tracking so periodic `releaseResources()` remains safe.

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

`Tensor` is a 2-D matrix abstraction backed by flat row-major `float[]`
storage — the core data type for every operation in DeepJ. All methods route through the active
`TensorBackend`, so the same code runs on CPU or GPU.

### Creating tensors

```java
// Zero-filled
Tensor a = new Tensor(3, 4);              // 3 rows × 4 cols, all zeros

// From data
Tensor b = Tensor.from2D(new float[][]{
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
a.data   // raw flat float[] row-major storage (trigger materialize() first if on GPU)
```

### Operations by category

#### Element-wise binary (same shape required)

```java
a.add(b)        a.subtract(b)        a.multiply(b)        a.divide(b)
a.matmul(b)     // [m×k] · [k×n] → [m×n]
```

#### Scalar

```java
a.multiplyScalar(2.0f)     a.addScalar(1.0f)     a.divideScalar(3.0f)
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
a.pow(2.0f)  a.clamp(0.0f, 1.0f)    a.transpose()
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

Use in-place selectively: it is ideal for local gradient accumulation/scaling,
but keep out-of-place ops for residual/cached/branch-shared tensors where
aliasing can affect correctness.

### GPU materialization

When the Metal backend is active, `data[]` may be stale. Call
`materialize()` before reading raw data, or use accessor methods
(`get`, `set`, `getRow`, `sum`, `print`) which materialise automatically:

```java
a.materialize();         // flush GPU ops, download result
float v = a.get(0, 0); // auto-materialises
a.print("my tensor");   // auto-materialises
```


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
Tensor x = Tensor.from2D(new float[][]{
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
});

Tensor y = Tensor.from2D(new float[][]{
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
        AdamW.defaultAdamW(3e-3f), // lr
        x,
        y,
        123L                      // seed
);

trainer.train(
        3000, // maxSteps         – stop after 3 000 gradient updates
        3,    // batchSize        – rows sampled per step
        200,  // logEvery         – print loss every 200 steps
        0.98f, // emaBeta          – smoothing factor for the moving-average loss
        1e-6f  // targetEmaLoss    – early-stop when smoothed loss falls below this
);
```

### Transformer stack

Each architecture has its own builder with only the fields it needs:

```java
// GPT-style — LayerNorm + MHSA + GELU FFN
TransformerStack gptStack = new GPTTransformerBuilder()
        .dModel(128).nHeads(4).dFF(512).nLayers(4)
        .ffnActivation(GELU::new)
        .seed(42)
        .build();

// Llama-style — RMSNorm + RoPE attention + SwiGLU
TransformerStack llamaStack = new LlamaTransformerBuilder()
        .dModel(512).nHeads(8).dFF(1408).nLayers(6)
        .maxSeqLen(2048)     // required: RoPE table size
        .seed(42)
        .build();

// DeepSeek-style — RMSNorm + MLA + SwiGLU (8× smaller KV cache)
TransformerStack deepSeekStack = new DeepSeekTransformerBuilder()
        .dModel(512).nHeads(8).dFF(1408).nLayers(6)
        .maxSeqLen(2048)     // required: RoPE table size
        .qRank(256)          // required: Q latent dim (e.g. dModel/2)
        .kvRank(128)         // required: KV latent dim (e.g. dModel/4)
        .seed(42)
        .build();
```

Each stack implements `Layer`, so you can call `forward()` / `backward()`
and collect `parameters()` like any other layer.

You can also construct individual blocks directly:

```java
// GPT-style block (LayerNorm + MHSA + configurable FFN)
GPTTransformerBlock block = new GPTTransformerBlock(
        512,       // dModel
        8,         // nHeads
        2048,      // dFF
        GELU::new, // FFN activation
        rnd
);

Tensor out = block.forward(x);   // [seqLen x dModel]
Tensor dX  = block.backward(grad);
```


### GPT training + generation

Train a small GPT-style language model.

```java
Tensor.setBackend(new MetalBackend());

Tokenizer tok = new ByteTokenizer();
TextDataset ds = TextDataset.fromFile(Path.of("sample_data/llm_training_dataset_1227_examples.txt"), tok, 256, 123);

GPTConfig cfg = new GPTConfig(
        tok.vocabSize(),
        256,         // maxSeqLen
        512,         // dModel
        4,           // nHeads
        5,           // nLayers
        1024         // dFF
);

GPTModel model = new GPTModel(cfg, 42);
Trainer trainer = CausalLMTraining.trainer(model, ds, 1e-4f);
trainer.train(
        10_000_000, // maxSteps
        2,          // batchSize
        1,          // logEvery
        0.98f,      // emaBeta
        0.01f,      // targetEmaLoss
        25,         // releaseEverySteps
        (step, loss, ema) -> {
            if (step > 0 && step % 500 == 0)
                model.save(Path.of("checkpoints/small-gpt-" + step + ".bin"));
        });

model.save(Path.of("checkpoints/small-gpt-final.bin"));

String out = TextGenerator.generate(
        model,            // model
        tok,              // tokenizer
        cfg,              // config
        "Bob Marley was ", // prompt
        200,              // maxNewTokens
        0.1f,             // temperature
        20,               // topK
        1234L             // seed
);
System.out.println(out);
```

### Llama training + generation

```java
Tensor.setBackend(new MetalBackend());

Tokenizer tok = new ByteTokenizer();
TextDataset ds = TextDataset.fromFile(Path.of("sample_data/corpus.txt"), tok, 256, 123);

LlamaConfig cfg = new LlamaConfig(
        tok.vocabSize(),
        256,                           // maxSeqLen
        512,                           // dModel
        4,                             // nHeads
        5,                             // nLayers
        LlamaConfig.defaultDFF(512)    // ≈ 1408
);

LlamaModel model = new LlamaModel(cfg, 42);
Trainer trainer = CausalLMTraining.trainer(model, ds, 1e-4f);

trainer.train(
        10_000_000, // maxSteps
        2,          // batchSize
        1,          // logEvery
        0.98f,      // emaBeta
        0.01f,      // targetEmaLoss
        25,         // releaseEverySteps
        (step, loss, ema) -> {
            if (step > 0 && step % 500 == 0)
                model.save(Path.of("checkpoints/small-llama-" + step + ".bin"));
        });

model.save(Path.of("checkpoints/small-llama-final.bin"));

String out = TextGenerator.generate(
        model,            // model
        tok,              // tokenizer
        cfg,              // config
        "Bob Marley was ", // prompt
        200,              // maxNewTokens
        0.1f,             // temperature
        20,               // topK
        1234L             // seed
);
System.out.println(out);
```

### DeepSeek training + generation

```java
Tensor.setBackend(new MetalBackend());

Tokenizer tok = new ByteTokenizer();
TextDataset ds = TextDataset.fromFile(Path.of("sample_data/corpus.txt"), tok, 256, 123);

DeepSeekConfig cfg = new DeepSeekConfig(
        tok.vocabSize(),
        256,         // maxSeqLen
        512,         // dModel
        4,           // nHeads
        5,           // nLayers
        1024,        // dFF
        512 / 2,     // qRank  — Q latent dimension
        512 / 4      // kvRank — KV latent dimension (8× smaller KV cache)
);

DeepSeekModel model = new DeepSeekModel(cfg, 42);
Trainer trainer = CausalLMTraining.trainer(model, ds, 1e-4f);

trainer.train(
        10_000_000, // maxSteps
        2,          // batchSize
        1,          // logEvery
        0.98f,      // emaBeta
        0.01f,      // targetEmaLoss
        25,         // releaseEverySteps
        (step, loss, ema) -> {
            if (step > 0 && step % 500 == 0)
                model.save(Path.of("checkpoints/small-deepseek-" + step + ".bin"));
        });

model.save(Path.of("checkpoints/small-deepseek-final.bin"));

String out = TextGenerator.generate(
        model,            // model
        tok,              // tokenizer
        cfg,              // config
        "Bob Marley was ", // prompt
        200,              // maxNewTokens
        0.1f,             // temperature
        20,               // topK
        1234L             // seed
);
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
| `RMSNorm1D` | RMS norm — no mean subtraction, no β; used by Llama / Mistral / Qwen / DeepSeek |
| `MultiHeadSelfAttention` | Multi-head causal self-attention (`[seqLen × dModel]`). Extensible via template-method hooks |
| `RoPEMultiHeadSelfAttention` | Extends `MultiHeadSelfAttention` with Rotary Positional Embedding (RoPE) |
| `MultiHeadLatentAttention` | MLA — compresses Q/K/V through low-rank bottlenecks; used by DeepSeek-V2/V3/R1 |
| `GPTTransformerBlock` | Pre-LN GPT-style block: `x + Attn(LN(x))`, then `x + FFN(LN(x))` |
| `LlamaTransformerBlock` | Pre-LN Llama-style block: RMSNorm + RoPE attention + SwiGLU |
| `DeepSeekTransformerBlock` | Pre-LN DeepSeek-style block: RMSNorm + MLA + SwiGLU |
| `SwiGLULayer` | Gated FFN: `down(SiLU(gate(x)) · up(x))` — used by Llama / Mistral / Qwen / DeepSeek |

Use individually or via `GPTTransformerBuilder`:

```java
// GPT-style block — LayerNorm + MHSA + configurable FFN
GPTTransformerBlock gptBlock = new GPTTransformerBlock(
        512,       // dModel
        8,         // nHeads
        2048,      // dFF
        GELU::new, // FFN activation
        rnd
);

// Llama-style block — RMSNorm + RoPE-MHSA + SwiGLU
LlamaTransformerBlock llamaBlock = new LlamaTransformerBlock(
        512,   // dModel
        8,     // nHeads
        1408,  // dFF  (≈ 8/3 × dModel, typical Llama ratio)
        2048,  // maxSeqLen — RoPE table size
        rnd
);

// DeepSeek-style block — RMSNorm + MLA + SwiGLU (8× smaller KV cache)
DeepSeekTransformerBlock dsBlock = new DeepSeekTransformerBlock(
        512,   // dModel
        8,     // nHeads
        256,   // qRank  — Q latent dimension (e.g. dModel/2)
        128,   // kvRank — KV latent dimension (e.g. dModel/4); only cKV is cached
        1408,  // dFF
        2048,  // maxSeqLen — RoPE table size
        rnd
);

Tensor out = dsBlock.forward(x);    // [seqLen x dModel]
Tensor dX  = dsBlock.backward(grad);
```

### Extending `MultiHeadSelfAttention`

`MultiHeadSelfAttention` is open for extension via two protected template-method
hooks. Override them to apply any Q/K transform without touching the base class:

```java
public class MyAttention extends MultiHeadSelfAttention {

    @Override
    protected Tensor transformQueryKey(Tensor heads, int seqLen) {
        return myTransform(heads, seqLen, nHeads); // forward transform
    }

    @Override
    protected Tensor transformQueryKeyBackward(Tensor gradHeads, int seqLen) {
        return myInverseTransform(gradHeads, seqLen, nHeads); // inverse
    }
}
```

`RoPEMultiHeadSelfAttention` is the built-in example — it overrides only
these two methods, inheriting all attention mechanics unchanged.

---

## 🔧 Transformer Builders / `TransformerStack`

Each architecture has a dedicated builder with only the fields it needs.
All builders produce a `TransformerStack` — a sequential chain of blocks
implementing `Layer`.

### `GPTTransformerBuilder`

| Method | Default | Description |
|---|---|---|
| `.dModel(int)` | required | Model / embedding dimension |
| `.nHeads(int)` | required | Number of attention heads |
| `.dFF(int)` | required | Feed-forward inner dimension |
| `.nLayers(int)` | required | Number of decoder blocks |
| `.ffnActivation(Supplier)` | `GELU::new` | Activation inside each FFN |
| `.seed(long)` | `42` | Weight initialisation seed |
| `.random(Random)` | — | Provide your own `Random` (overrides seed) |

### `LlamaTransformerBuilder`

| Method | Default | Description |
|---|---|---|
| `.dModel(int)` | required | Model / embedding dimension |
| `.nHeads(int)` | required | Number of attention heads |
| `.dFF(int)` | required | SwiGLU intermediate dimension |
| `.nLayers(int)` | required | Number of decoder blocks |
| `.maxSeqLen(int)` | required | RoPE table size |
| `.seed(long)` | `42` | Weight initialisation seed |
| `.random(Random)` | — | Provide your own `Random` (overrides seed) |

### `DeepSeekTransformerBuilder`

| Method | Default | Description |
|---|---|---|
| `.dModel(int)` | required | Model / embedding dimension |
| `.nHeads(int)` | required | Number of attention heads |
| `.dFF(int)` | required | SwiGLU intermediate dimension |
| `.nLayers(int)` | required | Number of decoder blocks |
| `.maxSeqLen(int)` | required | RoPE table size |
| `.qRank(int)` | required | Q latent dim for MLA (e.g. `dModel/2`) |
| `.kvRank(int)` | required | KV latent dim for MLA (e.g. `dModel/4`) — only `cKV` cached at inference |
| `.seed(long)` | `42` | Weight initialisation seed |
| `.random(Random)` | — | Provide your own `Random` (overrides seed) |

### TransformerStack

`TransformerStack` is a `record` that implements `Layer`:

```java
TransformerStack gptStack = new GPTTransformerBuilder()
        .dModel(256).nHeads(4).dFF(1024).nLayers(6)
        .ffnActivation(GELU::new).seed(42)
        .build();

TransformerStack llamaStack = new LlamaTransformerBuilder()
        .dModel(512).nHeads(8).dFF(1408).nLayers(6)
        .maxSeqLen(2048).seed(42)
        .build();

TransformerStack deepSeekStack = new DeepSeekTransformerBuilder()
        .dModel(512).nHeads(8).dFF(1408).nLayers(6)
        .maxSeqLen(2048).qRank(256).kvRank(128).seed(42)
        .build();

Tensor out = gptStack.forward(x);           // [seqLen x dModel]
Tensor dX  = gptStack.backward(gradOut);    // backprop through all blocks in reverse
List<Parameter> ps = gptStack.parameters(); // all block parameters
```

The stack is used internally by `GPTModel` and can also be used
standalone for custom architectures.

---

## 🤖 Models

All three model families share a common type hierarchy:

- **`CausalLM`** — interface declaring `forward(int[])`, `backward(Tensor)`, and `gradClipNorm()`. Any `CausalLM` works directly with `CausalLMTraining.trainer`.
- **`DecoderOnlyModel`** — abstract base class implementing the standard token-embedding → stack → norm → LM-head wiring. Subclasses supply their stack via the appropriate builder and override `gradClipNorm()`.

```
DecoderOnlyModel (implements CausalLM, Persistable)
├── GPTModel    — adds PositionalEmbedding + LayerNorm1D + initScale
├── LlamaModel  — RMSNorm1D, no positional embedding
└── DeepSeekModel — RMSNorm1D, MLA attention, no positional embedding
```

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
  → TransformerStack (nLayers × GPTTransformerBlock)
  → LayerNorm
  → Linear head → logits [seqLen × vocabSize]
```

### TextGenerator

Autoregressive sampling — works with `GPTModel`, `LlamaModel`, `DeepSeekModel`, or
any `Function<int[], Tensor>` forwarder:

```java
// GPT
String out = TextGenerator.generate(gptModel, tok, gptCfg, "Hello", 200, 0.8, 20, 42L);

// Llama
String out = TextGenerator.generate(llamaModel, tok, llamaCfg, "Hello", 200, 0.8, 20, 42L);

// DeepSeek
String out = TextGenerator.generate(deepSeekModel, tok, deepSeekCfg, "Hello", 200, 0.8, 20, 42L);

// Generic — any model that maps int[] → [seqLen × vocabSize]
String out = TextGenerator.generate(model::forward, cfg, tok, "Hello", 200, 0.8, 20, 42L);
```

Internally it runs an autoregressive loop: encode prompt → forward →
sample from top-k logits with temperature → append token → repeat.

---

### LlamaModel

Llama-style decoder-only transformer. Differences from GPT: no positional
embedding (RoPE handles position inside each attention block), `RMSNorm1D`
for the final norm, and `LlamaTransformerBlock` (RMSNorm + RoPE-Attn + SwiGLU)
instead of the classic GPT block.

```java
LlamaConfig cfg = new LlamaConfig(
        vocabSize,
        maxSeqLen,
        dModel,
        nHeads,
        nLayers,
        LlamaConfig.defaultDFF(dModel)  // ≈ 8/3 × dModel, rounded to multiple of 64
);

LlamaModel model = new LlamaModel(cfg, 42L);
```

#### LlamaConfig

| Field | Default | Description |
|---|---|---|
| `vocabSize` | required | Token vocabulary size |
| `maxSeqLen` | required | Maximum sequence length (RoPE table size) |
| `dModel` | required | Embedding / model dimension |
| `nHeads` | required | Attention heads (`dModel % nHeads == 0`) |
| `nLayers` | required | Number of transformer blocks |
| `dFF` | required | SwiGLU intermediate dimension |
| `gradClipNorm` | `1.0` | Global gradient clipping threshold |

`LlamaConfig.defaultDFF(dModel)` computes the standard Llama ratio
(`round(8/3 × dModel)` rounded to the nearest multiple of 64).

#### Architecture

```
inputIds
  → token embedding                        (no positional embedding — RoPE handles position)
  → nLayers × LlamaTransformerBlock
      x = x + RoPE-Attn( RMSNorm(x) )
      x = x + SwiGLU(    RMSNorm(x) )
  → RMSNorm
  → Linear head → logits [seqLen × vocabSize]
```

---

### DeepSeekModel

DeepSeek-style decoder-only transformer. The key difference from Llama is
**Multi-Head Latent Attention (MLA)**: Q and K/V are first compressed through
low-rank bottlenecks before attention is computed, dramatically reducing the
KV-cache footprint during inference.

```java
DeepSeekConfig cfg = new DeepSeekConfig(
        vocabSize,
        maxSeqLen,
        dModel,
        nHeads,
        nLayers,
        dFF,
        dModel / 2,   // qRank  — Q latent dimension
        dModel / 4    // kvRank — KV latent dimension (controls KV-cache size)
);

DeepSeekModel model = new DeepSeekModel(cfg, 42L);
```

#### DeepSeekConfig

| Field | Default | Description |
|---|---|---|
| `vocabSize` | required | Token vocabulary size |
| `maxSeqLen` | required | Maximum sequence length |
| `dModel` | required | Embedding / model dimension |
| `nHeads` | required | Attention heads |
| `nLayers` | required | Number of transformer blocks |
| `dFF` | required | SwiGLU intermediate dimension |
| `qRank` | required | Q latent dimension (e.g. `dModel/2`) |
| `kvRank` | required | KV latent dimension (e.g. `dModel/4`) — only `cKV` is cached during inference |
| `gradClipNorm` | `1.0` | Global gradient clipping threshold |

#### MLA — how it works

Standard MHA projects Q, K, V directly from `x` (shape `dModel × dModel` each).
MLA uses low-rank compression instead:

```
cQ  = x · Wdq     [seqLen × qRank]    Q  compression
Q   = cQ · Wuq    [seqLen × dModel]   Q  expansion  → RoPE → split heads

cKV = x · Wdkv    [seqLen × kvRank]   shared KV compression
K   = cKV · Wuk   [seqLen × dModel]   K  expansion  → RoPE → split heads
V   = cKV · Wuv   [seqLen × dModel]   V  expansion  → split heads
```

During inference only `cKV` (`seqLen × kvRank`) needs to be cached per layer —
not the full K and V tensors. With `kvRank = dModel/4` that is a 8× KV-cache
reduction vs standard MHA.

#### Architecture

```
inputIds
  → token embedding
  → nLayers × DeepSeekTransformerBlock
      x = x + MLA(    RMSNorm(x) )
      x = x + SwiGLU( RMSNorm(x) )
  → RMSNorm
  → Linear head → logits [seqLen × vocabSize]
```

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

// Convenience helper: reserves <BOS>, <EOS>, <PAD> automatically
BPETokenizer tok = new BPETrainer().trainTokenizerWithDefaults(corpusText, 1000);

// Versioned binary tokenizer persistence
BPEModelIO.save(Path.of("tokenizer.bpe"), tok.model());
BPETokenizer loadedTok = new BPETokenizer(BPEModelIO.load(Path.of("tokenizer.bpe")));

int[] ids  = loadedTok.encode("<BOS> hello world <EOS>");
String text = loadedTok.decode(ids);
```

---

## 📂 Data Loading

### TextDataset

Streaming, memory-mapped dataset that tokenises a text file and samples
random contiguous chunks for causal language model training.

The source text is streamed line-by-line (never loaded fully into heap),
tokenised in bounded chunks, and written to a temporary binary file.
That file is then memory-mapped so the OS pages token data in and out
on demand — datasets larger than available RAM work without changes.

```java
// From a file (streamed + memory-mapped — works for any file size)
TextDataset ds = TextDataset.fromFile(
        Path.of("corpus.txt"),
        tok,    // any Tokenizer
        256,    // seqLen – context window length
        123     // seed
);

// Or from pre-tokenised ids (small data / tests)
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
        AdamW.defaultAdamW(1e-3f), // optimizer
        x, y,                     // training data
        42L                       // seed
);
```

### CausalLMTraining

Factory for autoregressive language models. Accepts any `CausalLM` — `GPTModel`,
`LlamaModel`, `DeepSeekModel`, or your own implementation. Each step:

1.  `zeroGrad()`
2.  Sample a `Batch` from `TextDataset`
3.  For each sequence in the batch: `forward()` → `crossEntropyLoss()` → `backward()`
4.  Average gradients across the batch
5.  Global gradient clipping (using `gradClipNorm()`)
6.  `opt.step(parameters())`

Implementation note: gradient averaging/clipping scales gradients in-place,
and `Parameter.zeroGrad()` reuses existing CPU gradient buffers when safe
(falls back to fresh allocation for shape mismatches or GPU-tagged grads).
Loss/activation utility paths (`MSELoss`, `Sigmoid`, `Tanh`, `SiLU`) also
use in-place ops on temporary tensors where ownership is local.

```java
// Works for any CausalLM
Trainer gptTrainer      = CausalLMTraining.trainer(gptModel,      dataset, 1e-4f);
Trainer llamaTrainer    = CausalLMTraining.trainer(llamaModel,    dataset, 1e-4f);
Trainer deepSeekTrainer = CausalLMTraining.trainer(deepSeekModel, dataset, 1e-4f);
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
AdamW opt = AdamW.defaultAdamW(1e-3f);

// Full control
AdamW opt = new AdamW(1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f);

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

TransformerStack stack = new GPTTransformerBuilder()
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
`TensorBackend` operations are recorded into a lazy Metal compute graph and
executed in batches on flush/materialization boundaries. CPU ↔ GPU transfer is
automatic and only happens when CPU-side access requires synchronization.

### Enabling the backend

```java
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;

MetalBackend metal = new MetalBackend();
Tensor.setBackend(metal);
```

That's it — all `Tensor` operations route through the same backend entrypoint.
Use one backend instance consistently for a given execution path so op
recording and materialization share the same compute graph ownership.

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
 │  read e.data[]                 │
```

Key points:

-   **No round-trips (GPU-supported chains):** intermediate results (`c`, `d`) stay on GPU
-   **Batched execution:** all recorded ops flush in a single GPU command buffer
-   **Materialization on demand:** call `materialize()` (or use accessors like `get`, `set`, `print`) before reading CPU data
-   **Staleness tracking:** each tensor knows whether its CPU or GPU copy is current via `GpuBuffer.cpuStale` / `needsUpload` flags
-   **Zero-copy reuse:** a tensor's GPU buffer persists across operations — subsequent ops reuse it without re-uploading
-   **Backend ownership safety:** keep op recording and materialization on the same backend instance

A full forward + backward pass records hundreds of GPU ops and executes them
in just 2–3 native calls, minimising driver overhead.

---

## ⚡ Metal GPU Performance

Chained-pipeline benchmarks on a **2024 × 2024** matrix (4,096,576 elements),
measured on Apple Silicon with `-Dperf.iters.cpu=3` and `-Dperf.iters.gpu=10`.
Multiple lazy GPU ops are recorded into one command buffer, then a single
`materialize()` flushes them all — matching how the GPU runs during training.
CPU timings use the default multithreaded `DeepJExecutor` (parallel enabled).

| Pipeline | CPU (ms) | GPU (ms) | Speedup |
|---|---:|---:|---:|
| 5 mixed ops (1 matmul) | 374.922 | 14.088 | 26.61× |
| 10 mixed ops (2 matmuls) | 760.566 | 25.294 | 30.07× |
| 20 mixed ops (4 matmuls) | 4463.425 | 71.673 | 62.28× |
| 50 mixed ops (10 matmuls) | 9918.657 | 158.317 | 62.65× |
| linear fwd (3 ops) | 866.960 | 17.032 | 50.90× |
| attention fwd (4 ops) | 1709.401 | 25.236 | 67.74× |
| fwd + loss grad (6 ops) | 1759.048 | 34.268 | 51.33× |
| backward (6 ops, 2 matmuls) | 1769.151 | 31.267 | 56.58× |
| mini train step (9 ops) | 1752.929 | 28.748 | 60.98× |
| full train step (13 ops) | 3506.517 | 69.564 | 50.41× |

> **Key takeaway:** the lazy compute graph batches many GPU kernels into a
> single command buffer. As chain depth grows, speedups climb, with **peak
> measured speedup reaching ~68×** on Metal vs multithreaded CPU.

Run the benchmark yourself:

```bash
mvn test -Dtest=MetalBackendAllOpsPerformanceTest \
    '-Djunit.jupiter.conditions.deactivate=org.junit.jupiter.engine.extension.DisabledCondition' \
    -Dperf.size=2024 -Dperf.iters.cpu=3 -Dperf.iters.gpu=10 -Dperf.inplace.steps=100 \
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
