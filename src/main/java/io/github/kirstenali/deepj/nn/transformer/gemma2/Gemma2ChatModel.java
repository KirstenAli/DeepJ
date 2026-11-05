package io.github.kirstenali.deepj.nn.transformer.gemma2;

import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.nn.transformer.ConversationTurn;
import io.github.kirstenali.deepj.nn.transformer.TextGenerator;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tokenizers.gemma.Gemma2Chat;
import io.github.kirstenali.deepj.tokenizers.gemma.GemmaTokenizer;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Consumer;

/** Shared Gemma chat formatting, context fitting and streaming for CLI and desktop callers. */
public record Gemma2ChatModel(Gemma2 model, GemmaTokenizer tokenizer) {
    public Gemma2ChatModel {
        java.util.Objects.requireNonNull(model);
        java.util.Objects.requireNonNull(tokenizer);
        if (tokenizer.vocabSize() > model.config().vocabSize()) {
            throw new IllegalArgumentException("Gemma tokenizer exceeds the model vocabulary");
        }
    }

    public static Gemma2ChatModel load(Path selection, DType dtype) throws IOException {
        Path directory = Gemma2Snapshot.directory(selection);
        var tokenizer = new GemmaTokenizer(Gemma2Snapshot.tokenizer(directory));
        return new Gemma2ChatModel(Gemma2Pretrained.load(directory, dtype), tokenizer);
    }

    public String stream(String prompt, List<? extends ConversationTurn> history, int maxTokens, float temperature,
                         int topK, long seed, float penalty, ContextWindowMode mode, Consumer<String> listener) {
        String formatted = fit(prompt, history, maxTokens);
        String prefix = tokenizer.decode(tokenizer.encode(formatted));
        try (var session = model.openGeneration(mode)) {
            String result = TextGenerator.stream(session, model.config().maxSeqLen(), tokenizer, formatted,
                    maxTokens, temperature, topK, seed, penalty,
                    text -> listener.accept(continuation(prefix, text)));
            return continuation(prefix, result);
        }
    }

    private String fit(String prompt, List<? extends ConversationTurn> history, int maxTokens) {
        int window = model.config().maxSeqLen(), budget = window - Math.min(maxTokens, window / 2);
        String fitted = Gemma2Chat.format(prompt, List.of());
        if (tokenizer.encode(fitted).length > budget) {
            throw new IllegalArgumentException("Gemma prompt is too long; shorten it or clear history");
        }
        for (int first = history.size() - 1; first >= 0; first--) {
            String candidate = Gemma2Chat.format(prompt, history.subList(first, history.size()));
            if (tokenizer.encode(candidate).length > budget) break;
            fitted = candidate;
        }
        return fitted;
    }

    private static String continuation(String prefix, String text) {
        return text.startsWith(prefix) ? text.substring(prefix.length()) : text;
    }
}
