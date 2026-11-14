package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.nn.transformer.ConversationTurn;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tokenizers.qwen3.Qwen3Chat;
import io.github.kirstenali.deepj.tokenizers.qwen3.Qwen3Tokenizer;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;

/** Shared CLI/UI generation adapter using Qwen's thinking and non-thinking chat templates. */
public record Qwen3ChatModel(Qwen3 model, Qwen3Tokenizer tokenizer, boolean thinking) {
    public Qwen3ChatModel(Qwen3 model, Qwen3Tokenizer tokenizer) { this(model, tokenizer, false); }

    public Qwen3ChatModel withThinking(boolean enabled) { return new Qwen3ChatModel(model, tokenizer, enabled); }

    public Qwen3ChatModel {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tokenizer, "tokenizer");
        if (tokenizer.vocabSize() > model.config().vocabSize()) {
            throw new IllegalArgumentException("Tokenizer IDs exceed the model vocabulary");
        }
    }

    public static Qwen3ChatModel load(Path selection) throws IOException {
        return load(selection, null);
    }

    /** Chat follows checkpoint precision for weights and embedding activations; an explicit dtype overrides both. */
    public static Qwen3ChatModel load(Path selection, DType dtype) throws IOException {
        Path directory = Qwen3Snapshot.directory(selection);
        Qwen3Tokenizer tokenizer = Qwen3Tokenizer.load(directory);
        return new Qwen3ChatModel(Qwen3Pretrained.load(directory, dtype, true), tokenizer);
    }

    public String stream(String prompt, int maxTokens, float temperature, int topK, long seed,
                         float penalty, ContextWindowMode mode, Consumer<String> listener) {
        return support().stream(prompt, maxTokens, temperature, topK, seed, penalty, mode, listener);
    }

    public String streamConversation(List<Qwen3Chat.Message> messages, int maxTokens, float temperature, int topK,
                                     long seed, float penalty, ContextWindowMode mode, Consumer<String> listener) {
        return support().streamConversation(messages, maxTokens, temperature, topK, seed, penalty, mode, listener);
    }

    public String streamConversation(String prompt, List<? extends ConversationTurn> history, int maxTokens,
                                     float temperature, int topK, long seed, float penalty, ContextWindowMode mode,
                                     Consumer<String> listener) {
        return support().streamConversation(prompt, history, maxTokens, temperature, topK, seed,
                penalty, mode, listener);
    }

    private Qwen3ChatSupport support() {
        return new Qwen3ChatSupport(model.config(), tokenizer, thinking, model::openGeneration);
    }
}
