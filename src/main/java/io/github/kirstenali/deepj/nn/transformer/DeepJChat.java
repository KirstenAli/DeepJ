package io.github.kirstenali.deepj.nn.transformer;

import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.function.Consumer;

public final class DeepJChat {

    private static final int DEFAULT_MAX_TOKENS = 80;
    private static final float DEFAULT_TEMPERATURE = 0.8f;
    private static final int DEFAULT_TOP_K = 40;
    private static final long DEFAULT_SEED = 1234L;

    private final DecoderOnlyModel model;
    private final Tokenizer tokenizer;
    private final ContextWindowMode contextWindow;

    private DeepJChat(DecoderOnlyModel model, Tokenizer tokenizer) {
        this(model, tokenizer, ContextWindowMode.REBUILD);
    }

    private DeepJChat(DecoderOnlyModel model, Tokenizer tokenizer, ContextWindowMode contextWindow) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.contextWindow = contextWindow;
    }

    /** Returns a view sharing this model and tokenizer, with independent request-scoped caches. */
    public DeepJChat withContextWindow(ContextWindowMode mode) {
        Objects.requireNonNull(mode, "mode");
        if (mode == ContextWindowMode.ROLLING && !model.supportsRollingGeneration()) {
            throw new UnsupportedOperationException("This model does not support rolling cached generation");
        }
        return new DeepJChat(model, tokenizer, mode);
    }

    public static DeepJChat load(Path path) throws IOException {
        Objects.requireNonNull(path, "path");
        Path file = Files.isDirectory(path) ? path.resolve("model.dj") : path;
        DecoderOnlyModel model = ModelLoader.load(file);
        Path tokenizerPath = file.toAbsolutePath().getParent().resolve("tokenizer.bpe");
        Tokenizer tokenizer = tokenizer(tokenizerPath, model.config().vocabSize());
        if (model.config().vocabSize() != tokenizer.vocabSize()) {
            throw new IOException("Tokenizer vocabulary does not match model");
        }
        return new DeepJChat(model, tokenizer);
    }

    private static Tokenizer tokenizer(Path path, int vocabSize) throws IOException {
        if (Files.isRegularFile(path)) return new BPETokenizer(BPEModelIO.load(path));
        if (vocabSize == ByteTokenizer.VOCAB_SIZE) return new ByteTokenizer();
        throw new IOException("Missing tokenizer.bpe: " + path);
    }

    public String generate(String prompt) {
        return generate(prompt, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_SEED);
    }

    public String generate(String prompt, int maxTokens, float temperature, int topK, long seed) {
        return generate(prompt, maxTokens, temperature, topK, seed, 1.0f);
    }

    public String generate(String prompt, int maxTokens, float temperature, int topK,
                           long seed, float repetitionPenalty) {
        String formatted = model.promptFormat().apply(prompt);
        String prefix = tokenizer.decode(tokenizer.encode(formatted));
        try (var session = model.openGeneration(contextWindow)) {
            String result = TextGenerator.generate(session, model.config().maxSeqLen(), tokenizer,
                    formatted, maxTokens, temperature, topK, seed, repetitionPenalty);
            return continuation(prefix, result);
        }
    }

    public String stream(String prompt, Consumer<String> listener) {
        return stream(prompt, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_K,
                DEFAULT_SEED, listener);
    }

    public String stream(String prompt, int maxTokens, float temperature, int topK,
                         long seed, Consumer<String> listener) {
        return stream(prompt, maxTokens, temperature, topK, seed, 1.0f, listener);
    }

    public String stream(String prompt, int maxTokens, float temperature, int topK,
                         long seed, float repetitionPenalty, Consumer<String> listener) {
        Objects.requireNonNull(listener, "listener");
        String formatted = model.promptFormat().apply(prompt);
        String prefix = tokenizer.decode(tokenizer.encode(formatted));
        try (var session = model.openGeneration(contextWindow)) {
            String result = TextGenerator.stream(session, model.config().maxSeqLen(), tokenizer, formatted,
                    maxTokens, temperature, topK, seed, repetitionPenalty,
                    text -> listener.accept(continuation(prefix, text)));
            return continuation(prefix, result);
        }
    }

    private static String continuation(String prefix, String result) {
        return result.startsWith(prefix) ? result.substring(prefix.length()) : result;
    }
}
