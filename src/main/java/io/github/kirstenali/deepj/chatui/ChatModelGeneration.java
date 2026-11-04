package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.Agent;
import io.github.kirstenali.deepj.nn.transformer.DecoderOnlyModel;
import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.nn.transformer.PromptFormat;
import io.github.kirstenali.deepj.nn.transformer.TextGenerator;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.util.List;
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
        return streamFormatted(model, tokenizer, model.promptFormat().apply(prompt), maxTokens, temperature, topK,
                seed, repetitionPenalty, mode, listener);
    }

    static Agent.Turn streamTurn(DecoderOnlyModel model, Tokenizer tokenizer, String prompt, List<Agent.Turn> history,
                                 ChatGenerationSettings settings, Consumer<String> listener) {
        String formatted = conversationPrompt(model.promptFormat(), tokenizer, prompt, history,
                model.config().maxSeqLen(), settings.maxTokens());
        String reply = streamFormatted(model, tokenizer, formatted, settings.maxTokens(),
                (float) settings.temperature(), settings.topK(), settings.seed(), (float) settings.repetitionPenalty(),
                settings.contextWindow(), listener);
        return Agent.Turn.answered(prompt, reply);
    }

    static String conversationPrompt(PromptFormat format, Tokenizer tokenizer, String prompt,
                                     List<Agent.Turn> history, int window, int maxTokens) {
        String transcript = exchange(format, prompt);
        String fitted = format.apply(prompt);
        int budget = window - Math.min(maxTokens, window / 2);
        for (int index = history.size() - 1; index >= 0; index--) {
            var turn = history.get(index);
            String candidate = exchange(format, turn.prompt()) + "\n" + turn.reply() + "\n\n" + transcript;
            if (tokenizer.encode(candidate).length > budget) break;
            transcript = candidate;
            fitted = candidate;
        }
        return fitted;
    }

    private static String exchange(PromptFormat format, String prompt) {
        return format == PromptFormat.PLAIN ? "User:\n" + prompt + "\nAssistant:" : format.apply(prompt);
    }

    private static String streamFormatted(DecoderOnlyModel model, Tokenizer tokenizer, String formatted,
                                          int maxTokens, float temperature, int topK, long seed,
                                          float repetitionPenalty, ContextWindowMode mode, Consumer<String> listener) {
        String prefix = tokenizer.decode(tokenizer.encode(formatted));
        try (var session = model.openGeneration(mode)) {
            String result = TextGenerator.stream(session, model.config().maxSeqLen(), tokenizer, formatted,
                    maxTokens, temperature, topK, seed, repetitionPenalty,
                    text -> listener.accept(GeneratedText.continuation(prefix, text)));
            return GeneratedText.continuation(prefix, result);
        }
    }
}
