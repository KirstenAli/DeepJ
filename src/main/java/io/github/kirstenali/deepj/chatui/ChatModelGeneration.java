package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.transformer.DecoderOnlyModel;
import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.nn.transformer.TextGenerator;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.util.function.Consumer;

final class ChatModelGeneration {

    private ChatModelGeneration() {}

    static String stream(DecoderOnlyModel model, Tokenizer tokenizer, String prompt,
                         int maxTokens, float temperature, int topK, long seed,
                         Consumer<String> listener) {
        return stream(model, tokenizer, prompt, maxTokens, temperature, topK, seed, 1.0f, listener);
    }

    static String stream(DecoderOnlyModel model, Tokenizer tokenizer, String prompt,
                         int maxTokens, float temperature, int topK, long seed,
                         float repetitionPenalty, Consumer<String> listener) {
        return stream(model, tokenizer, prompt, maxTokens, temperature, topK, seed,
                repetitionPenalty, ContextWindowMode.REBUILD, listener);
    }

    static String stream(DecoderOnlyModel model, Tokenizer tokenizer, String prompt,
                         int maxTokens, float temperature, int topK, long seed, float repetitionPenalty,
                         ContextWindowMode mode, Consumer<String> listener) {
        String formatted = model.promptFormat().apply(prompt);
        String prefix = tokenizer.decode(tokenizer.encode(formatted));
        try (var session = model.openGeneration(mode)) {
            String result = TextGenerator.stream(session, model.config().maxSeqLen(), tokenizer, formatted,
                    maxTokens, temperature, topK, seed, repetitionPenalty,
                    text -> listener.accept(GeneratedText.continuation(prefix, text)));
            return GeneratedText.continuation(prefix, result);
        }
    }
}
