package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.models.TextGenerator;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.util.function.Consumer;

final class ChatModelGeneration {

    private ChatModelGeneration() {}

    static String stream(DecoderOnlyModel model, Tokenizer tokenizer, String prompt,
                         int maxTokens, float temperature, int topK, long seed,
                         Consumer<String> listener) {
        String formatted = model.promptFormat().apply(prompt);
        String prefix = tokenizer.decode(tokenizer.encode(formatted));
        String result = TextGenerator.stream(model, tokenizer, model.config(), formatted,
                maxTokens, temperature, topK, seed,
                text -> listener.accept(GeneratedText.continuation(prefix, text)));
        return GeneratedText.continuation(prefix, result);
    }
}
