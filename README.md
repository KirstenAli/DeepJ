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
        <version>0.4.0-alpha</version>
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
        10_000_000,    // maxSteps
        2,             // batchSize
        1,             // logEvery
        0.98,          // emaBeta
        0.01,          // targetEmaLoss
        25,            // releaseEverySteps – free orphaned GPU buffers every N steps
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
        200,           // maxNewTokens
        0.1,           // temperature
        20,            // topK
        1234L          // seed
);

System.out.println("\n=== Generated ===");
System.out.println(out);
```

## 🖥️ Metal GPU Backend (macOS)

DeepJ ships with an optional **Metal GPU backend** for macOS Apple Silicon.
All GPU-capable tensor operations always dispatch to Metal — there are no
size thresholds or CPU fallbacks for those ops. CPU ↔ GPU data transfer is
handled automatically and only happens when necessary.

### Enabling the backend

```java
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;

MetalBackend metal = new MetalBackend();
Tensor.setBackend(metal);
```

That's it — every supported `Tensor` operation now routes through Metal.
Operations without a GPU kernel (broadcasts, reductions, indexing) continue
to run on the CPU and materialise automatically as needed.

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
-   **CPU-only ops** (broadcasts, reductions, indexing) materialise tensors on demand and delegate to the CPU backend

A full forward + backward pass records hundreds of GPU ops and executes them
in just 2–3 native calls, minimising driver overhead.

## ⚡ Metal GPU Performance


Below are CPU vs GPU timings for all tensor operations on a
**512 x 512** matrix (262,144 elements), measured on Apple Silicon
with `-Dperf.iters.cpu=3` and `-Dperf.iters.gpu=10`.

| Operation | CPU (ms) | GPU (ms) | Speedup |
|---|---:|---:|---:|
| matmul | 23.348 | 0.160 | 146.04x |
| matmul (rect) | 7.441 | 0.123 | 60.36x |
| add | 0.303 | 0.087 | 3.48x |
| subtract | 0.403 | 0.166 | 2.43x |
| multiply | 0.336 | 0.053 | 6.35x |
| divide | 0.287 | 0.047 | 6.18x |
| multiplyScalar | 0.441 | 0.064 | 6.92x |
| addScalar | 0.335 | 0.226 | 1.48x |
| divideScalar | 0.413 | 0.142 | 2.91x |
| addRowVector | 0.326 | 0.140 | 2.33x |
| addBroadcastCols | 0.132 | 0.144 | 0.91x |
| subtractBroadcastCols | 0.339 | 0.119 | 2.85x |
| multiplyBroadcastCols | 0.128 | 0.132 | 0.97x |
| divideBroadcastCols | 0.179 | 0.129 | 1.39x |
| addBroadcastRows | 0.114 | 0.098 | 1.16x |
| multiplyBroadcastRows | 0.555 | 0.106 | 5.22x |
| sumRows | 0.490 | 0.034 | 14.63x |
| sumAlongRows | 0.327 | 0.078 | 4.18x |
| sumAlongCols | 0.034 | 0.034 | 1.00x |
| meanAlongRows | 0.124 | 0.105 | 1.18x |
| varianceAlongRows | 0.237 | 0.230 | 1.03x |
| maxAlongRows | 0.060 | 0.060 | 1.01x |
| sum (scalar) | 0.476 | 0.148 | 3.22x |
| sumAbs (scalar) | 0.152 | 0.151 | 1.01x |
| sqrt | 0.709 | 0.069 | 10.22x |
| pow (^2) | 0.241 | 0.232 | 1.04x |
| neg | 0.609 | 0.142 | 4.29x |
| exp | 0.446 | 0.059 | 7.59x |
| log | 0.405 | 0.055 | 7.40x |
| clamp | 0.126 | 0.122 | 1.03x |
| transpose | 0.159 | 0.158 | 1.00x |
| tanh | 0.631 | 0.168 | 3.75x |
| sigmoid | 0.478 | 0.048 | 9.95x |
| relu | 0.118 | 0.054 | 2.19x |
| reluBackward | 0.135 | 0.051 | 2.66x |
| gelu | 0.696 | 0.052 | 13.49x |
| geluBackward | 0.940 | 0.059 | 15.91x |
| softmaxRows | 1.721 | 0.049 | 35.20x |
| softmaxBackward | 1.043 | 0.054 | 19.23x |
| crossEntropyLoss | 0.511 | 0.357 | 1.43x |
| crossEntropyGradient | 1.010 | 0.067 | 14.97x |
| adamWUpdate | 1.066 | 0.001 | 1706.07x |
| layerNormBackward | 1.015 | 0.058 | 17.36x |
| sliceRows | 0.035 | 0.037 | 0.95x |
| scatterAddRows | 0.123 | 0.122 | 1.01x |

> **Key takeaway:** most high-throughput math ops see clear Metal speedups,
> while a few broadcast/indexing paths remain around parity (or slightly
> CPU-favored) depending on tensor shape and dispatch overhead.

Run the benchmark yourself:

```bash
mvn test -Dtest=MetalBackendAllOpsPerformanceTest \
    "-Djunit.jupiter.conditions.deactivate=org.junit.*" \
    -Dperf.size=512 -Dperf.iters.cpu=3 -Dperf.iters.gpu=10 \
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
