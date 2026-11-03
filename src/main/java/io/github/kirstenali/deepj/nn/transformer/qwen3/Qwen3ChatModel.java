package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.nn.transformer.TextGenerator;
import io.github.kirstenali.deepj.tokenizers.qwen3.Qwen3Chat;
import io.github.kirstenali.deepj.tokenizers.qwen3.Qwen3Tokenizer;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;

/** Shared CLI/UI generation adapter using Qwen's non-thinking text chat template. */
public record Qwen3ChatModel(Qwen3 model, Qwen3Tokenizer tokenizer) {
    public Qwen3ChatModel {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tokenizer, "tokenizer");
        if (tokenizer.vocabSize() > model.config().vocabSize()) {
            throw new IllegalArgumentException("Tokenizer IDs exceed the model vocabulary");
        }
    }

    public static Qwen3ChatModel load(Path selection) throws IOException {
        Path directory = Qwen3Snapshot.directory(selection);
        Qwen3Tokenizer tokenizer = Qwen3Tokenizer.load(directory);
        return new Qwen3ChatModel(Qwen3Pretrained.load(directory), tokenizer);
    }

    public String stream(String prompt, int maxTokens, float temperature, int topK, long seed,
                         float penalty, ContextWindowMode mode, Consumer<String> listener) {
        String formatted = Qwen3Chat.format(List.of(new Qwen3Chat.Message(Qwen3Chat.Role.USER, prompt)), true, false);
        String prefix = tokenizer.decode(tokenizer.encode(formatted));
        try (var session = model.openGeneration(mode)) {
            String text = TextGenerator.stream(session, model.config().maxSeqLen(), tokenizer, formatted,
                    maxTokens, temperature, topK, seed, penalty,
                    generated -> listener.accept(continuation(prefix, generated)));
            return continuation(prefix, text);
        }
    }

    private static String continuation(String prefix, String text) {
        return text.startsWith(prefix) ? text.substring(prefix.length()) : text;
    }
}
