# DeepJ

**DeepJ** is a lightweight Java library for building and experimenting
with **Transformer-based neural networks**.

The library focuses on:

-   simple APIs
-   minimal dependencies
-   easy experimentation
-   clean Java implementations of modern architectures

📚 Documentation\
https://kirstenali.github.io/DeepJ/

------------------------------------------------------------------------

# 🚀 Installation

Add the GitHub Packages repository and dependency to your `pom.xml`.

``` xml
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
        <version>0.1.10-alpha</version>
    </dependency>
</dependencies>
```

------------------------------------------------------------------------

# 📚 Examples

## Classic ANN-style MLP (FNN)

Train a small feed-forward network.

``` java
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
        3,
        new int[]{16, 16},
        3,
        GELU::new,
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
        3000,
        0.98,
        1e-6,
        200,
        3
);
```

------------------------------------------------------------------------

## Transformer stack builder

Create a stack of decoder blocks using the builder.

``` java
TransformerStack stack = new TransformerBuilder()
        .dModel(128)
        .nHeads(4)
        .dFF(512)
        .nLayers(4)
        .ffnActivation(GELU::new)
        .seed(42)
        .build();
```

------------------------------------------------------------------------

## Tiny GPT training + generation

Train a small GPT-style language model.

``` java
Path corpus = Path.of("sample_data/sample_corpus.txt");

Tokenizer tok = new ByteTokenizer();
TextDataset ds = TextDataset.fromFile(corpus, tok, 64, 123);

GPTConfig cfg = new GPTConfig(
        tok.vocabSize(),
        64,
        128,
        4,
        2,
        4 * 128
);

GPTModel model = new GPTModel(cfg, 42);

Trainer trainer = CausalLMTraining.trainer(model, ds, 1e-3);

trainer.train(
        10_000,
        16,
        50,
        0.98,
        0.25
);
```

### Generate text

``` java
String prompt = "Mara wrote down the rhythm, ";

String out = TextGenerator.generate(
        model,
        tok,
        cfg,
        prompt,
        64,
        0.1,
        0,
        1234L
);

System.out.println("\n=== Generated ===");
System.out.println(out);
```

------------------------------------------------------------------------

# 💬 Chat UI

DeepJ also includes a simple **JavaFX chat interface** that lets you
interact with trained models.

The UI itself is **model-agnostic**.\
You provide your own implementation of the `ChatService` interface.

This allows you to:

-   control how models are loaded
-   configure tokenizers
-   customize generation settings
-   plug in completely different model types

------------------------------------------------------------------------

# Using the Chat UI

To launch the UI, extend `BaseChatApp` and return your own
`ChatService`.

``` java
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

------------------------------------------------------------------------

# Implementing a ChatService

Your service controls:

-   model configuration
-   tokenizer choice
-   model loading
-   generation behaviour

Example implementation using DeepJ GPT:

``` java
public class GPTChatService implements ChatService {

    private GPTModel model;
    private final Tokenizer tokenizer = new ByteTokenizer();

    private final GPTConfig config = new GPTConfig(
            ByteTokenizer.VOCAB_SIZE,
            128,
            256,
            4,
            4,
            1024
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

------------------------------------------------------------------------

# Example UI Flow

1.  User selects a trained `.bin` model
2.  `ChatService.loadModel()` loads the model
3.  User enters a prompt
4.  UI calls `chatService.generate(...)`
5.  Generated text appears in the chat

------------------------------------------------------------------------

# 📄 License

MIT License.
