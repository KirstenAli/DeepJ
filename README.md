<p align="center">
  <img src="./deepj_logo.svg" alt="DeepJ logo" width="170" />
</p>

# DeepJ

A lightweight Java library for building and experimenting with
**Transformer-based neural networks**.

DeepJ focuses on:

-   simple APIs
-   minimal dependencies
-   easy experimentation
-   clean Java implementations of modern architectures

📚 **Documentation:** https://kirstenali.github.io/DeepJ/

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
        <version>0.3.0-alpha</version>
    </dependency>
</dependencies>
```

## 📚 Examples

### Classic ANN-style MLP (FNN)

Train a small feed-forward neural network.

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
        GELU::new,
        null,              // outputActivation
        rnd
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
        3000, // maxSteps
        3,    // batchSize
        200,  // logEvery
        0.98, // emaBeta
        1e-6  // targetEmaLoss
);
```

### Transformer stack builder

Create a stack of decoder blocks using the builder.

```java
TransformerStack stack = new TransformerBuilder()
        .dModel(128)
        .nHeads(4)
        .dFF(512)
        .nLayers(4)
        .ffnActivation(GELU::new)
        .seed(42)
        .build();
```

### Tiny GPT training + generation

Train a small GPT-style language model.

```java
Path corpus = Path.of("sample_data/sample_corpus.txt");

Tokenizer tok = new ByteTokenizer();
TextDataset ds = TextDataset.fromFile(
        corpus,
        tok,
        64,     // seqLen
        123     // seed
);

GPTConfig cfg = new GPTConfig(
        tok.vocabSize(),
        64,     // maxSeqLen
        128,    // dModel
        4,      // nHeads
        2,      // nLayers
        4 * 128, // dFF
        1.0,     // initScale
        1.0      // gradClipNorm
);

GPTModel model = new GPTModel(cfg, 42);

Trainer trainer = CausalLMTraining.trainer(
        model,
        ds,
        1e-3 // lr
);

trainer.train(
        10_000, // maxSteps
        16,     // batchSize
        50,     // logEvery
        0.98,   // emaBeta
        0.25,   // targetEmaLoss
        25      // releaseEverySteps (periodically free backend GPU/native resources)
);
```

Generate text:

```java
String prompt = "Mara wrote down the rhythm, ";

String out = TextGenerator.generate(
        model,
        tok,
        cfg,
        prompt,
        64,    // maxNewTokens
        0.1,   // temperature
        0,     // topK
        1234L  // seed
);

System.out.println("\n=== Generated ===");
System.out.println(out);
```

## 🖥️ Metal GPU Backend (macOS)

DeepJ ships with an optional **Metal GPU backend** for macOS Apple Silicon.
All tensor operations automatically dispatch to the GPU when the input is
large enough, with seamless CPU ↔ GPU data transfer.

### Enabling the backend

```java
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;

MetalBackend metal = new MetalBackend();
Tensor.setBackend(metal);

System.out.println("GPU available: " + MetalBackend.isGpuAvailable());
```

That's it — every `Tensor` operation now routes through Metal when beneficial.

### Tuning GPU dispatch thresholds

The backend only dispatches to the GPU when the workload is large enough
to offset the overhead. You can tune the thresholds:

```java
MetalBackend metal = new MetalBackend();

// Matmul: minimum m*k*n work units (default: 1,048,576)
metal.setMatmulGpuThreshold(128L * 128 * 64);

// Element-wise: minimum total elements (default: 4,096)
metal.setElementwiseGpuThreshold(4096);

// Set to 0 to force everything to GPU
metal.setMatmulGpuThreshold(0);
metal.setElementwiseGpuThreshold(0);

Tensor.setBackend(metal);
```

### Debugging dispatch decisions

Enable logging to see which ops go to CPU vs GPU:

```java
metal.setLogDispatches(true);
// prints to stderr:
// [Metal] matmul [256x256]·[256x128] work=8,388,608 → GPU
// [Metal] add [64x64] n=4,096 → GPU
// [Metal] sqrt [32x32] n=1,024 → CPU
```

### Falling back to CPU

If Metal is not available (non-macOS, no Apple Silicon), the backend
automatically falls back to CPU for every operation — no code changes needed.
You can also check availability at runtime:

```java
if (MetalBackend.isGpuAvailable()) {
    Tensor.setBackend(new MetalBackend());
} else {
    System.out.println("Metal not available, using CPU backend");
}
```

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

-   **No round-trips:** intermediate results (`c`, `d`) never leave the GPU
-   **Batched execution:** all recorded ops flush in a single GPU command buffer
-   **Automatic materialization:** reading `data[][]` or calling `materialize()` triggers the flush
-   **Staleness tracking:** each tensor knows whether its CPU or GPU copy is current via `GpuBuffer.cpuStale` / `needsUpload` flags
-   **Zero-copy reuse:** a tensor's GPU buffer persists across operations — subsequent ops reuse it without re-uploading

This means a full forward + backward pass can record hundreds of GPU ops
and execute them in just 2–3 native calls, minimising driver overhead.

## ⚡ Metal GPU Performance


Below are CPU vs GPU timings for all tensor operations on a
**512 × 512** matrix (262,144 elements), measured on an Apple M1.

| Operation | CPU (ms) | GPU (ms) | Speedup |
|---|---:|---:|---:|
| **matmul** | 24.701 | 0.051 | **481.18×** |
| **matmul (rect)** | 6.566 | 0.132 | **49.57×** |
| add | 0.435 | 0.048 | 9.15× |
| subtract | 0.170 | 0.048 | 3.52× |
| multiply | 0.243 | 0.051 | 4.79× |
| divide | 0.361 | 0.052 | 6.93× |
| multiplyScalar | 0.445 | 0.129 | 3.46× |
| addScalar | 0.205 | 0.185 | 1.11× |
| divideScalar | 0.204 | 0.194 | 1.05× |
| addRowVector | 0.189 | 0.175 | 1.08× |
| addBroadcastCols | 0.090 | 0.085 | 1.06× |
| subtractBroadcastCols | 0.112 | 0.086 | 1.30× |
| multiplyBroadcastCols | 0.085 | 0.087 | 0.98× |
| divideBroadcastCols | 0.088 | 0.089 | 0.99× |
| addBroadcastRows | 0.085 | 0.085 | 1.00× |
| multiplyBroadcastRows | 0.094 | 0.089 | 1.06× |
| sumRows | 0.219 | 0.034 | 6.44× |
| sumAlongRows | 0.083 | 0.064 | 1.29× |
| sumAlongCols | 0.033 | 0.033 | 1.01× |
| meanAlongRows | 0.072 | 0.069 | 1.04× |
| varianceAlongRows | 0.157 | 0.155 | 1.01× |
| maxAlongRows | 0.043 | 0.042 | 1.02× |
| sum (scalar) | 0.142 | 0.142 | 1.00× |
| sumAbs (scalar) | 0.153 | 0.151 | 1.01× |
| **sqrt** | 0.318 | 0.046 | **6.89×** |
| pow (^2) | 0.192 | 0.195 | 0.98× |
| neg | 0.134 | 0.139 | 0.96× |
| **exp** | 0.331 | 0.048 | **6.88×** |
| **log** | 0.299 | 0.051 | **5.88×** |
| clamp | 0.111 | 0.104 | 1.07× |
| transpose | 0.125 | 0.121 | 1.03× |
| **tanh** | 0.438 | 0.046 | **9.61×** |
| **sigmoid** | 0.303 | 0.053 | **5.69×** |
| relu | 0.185 | 0.048 | 3.81× |
| reluBackward | 0.111 | 0.050 | 2.24× |
| **gelu** | 0.507 | 0.049 | **10.39×** |
| **geluBackward** | 0.683 | 0.129 | **5.29×** |
| softmaxRows | 0.451 | 0.125 | 3.60× |
| softmaxBackward | 0.236 | 0.237 | 0.99× |
| crossEntropyLoss | 0.468 | 0.275 | 1.71× |
| crossEntropyGradient | 0.862 | 0.757 | 1.14× |
| adamWUpdate | 0.255 | 0.281 | 0.91× |
| layerNormBackward | 0.254 | 0.205 | 1.24× |
| sliceRows | 0.042 | 0.041 | 1.02× |
| scatterAddRows | 0.121 | 0.121 | 1.00× |

> **Key takeaway:** GPU-accelerated ops (matmul, activations, unary math)
> see **3×–481× speedups**. Ops that currently fall back to CPU show ~1×
> (no overhead from the dispatch layer).

Run the benchmark yourself:

```bash
mvn test -Dtest=MetalBackendAllOpsPerformanceTest \
    "-Djunit.jupiter.conditions.deactivate=org.junit.*" \
    -Dperf.size=512 -Dperf.iters.cpu=5 -Dperf.iters.gpu=10 \
    -Dsurefire.useFile=false
```

## 💬 Chat UI

DeepJ also includes an **optional JavaFX chat interface** for
interacting with trained models.

The UI is **model-agnostic**, meaning you provide your own
implementation of the `ChatService` interface.

This allows you to:

-   control how models are loaded
-   configure tokenizers
-   customize generation settings
-   plug in completely different model types

### Using the Chat UI

To launch the UI, extend `BaseChatApp` and provide your own
`ChatService`.

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

### Implementing a ChatService

Your service controls:

-   model configuration
-   tokenizer choice
-   model loading
-   generation behaviour

Example implementation using DeepJ GPT:

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
        model = new GPTModel(
                config,
                42      // seed
        );
        model.load(modelPath);
    }

    @Override
    public boolean isModelLoaded() {
        return model != null;
    }

    @Override
    public String generate(String prompt, int maxTokens, double temperature, int topK, long seed) {

        return TextGenerator.generate(
                model,
                tokenizer,
                config,
                prompt,
                maxTokens,
                temperature,
                topK,
                seed
        );
    }
}
```

### Example UI Flow

1.  User selects a trained `.bin` model
2.  `ChatService.loadModel()` loads the model
3.  User enters a prompt
4.  The UI calls `chatService.generate(...)`
5.  Generated text appears in the chat window

## 📄 License

MIT License.
