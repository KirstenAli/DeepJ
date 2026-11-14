package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.nn.transformer.ConversationTurn;
import io.github.kirstenali.deepj.nn.transformer.GenerationSession;
import io.github.kirstenali.deepj.nn.transformer.TextGenerator;
import io.github.kirstenali.deepj.tokenizers.qwen3.Qwen3Chat;
import io.github.kirstenali.deepj.tokenizers.qwen3.Qwen3Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

/** Shared chat prompts and sampling for dense and quantized Qwen3 sessions. */
public record Qwen3ChatSupport(Qwen3Config config, Qwen3Tokenizer tokenizer, boolean thinking,
                               Function<ContextWindowMode, GenerationSession> sessions) {
    public String stream(String prompt, int maxTokens, float temperature, int topK, long seed,
                         float penalty, ContextWindowMode mode, Consumer<String> listener) {
        String formatted = Qwen3Chat.format(List.of(
                new Qwen3Chat.Message(Qwen3Chat.Role.USER, prompt)), true, thinking);
        return streamFormatted(formatted, maxTokens, temperature, topK, seed, penalty, mode, listener);
    }

    /** Streams a conversation of user and assistant messages ending with the current user request. */
    public String streamConversation(List<Qwen3Chat.Message> messages, int maxTokens, float temperature, int topK,
                                     long seed, float penalty, ContextWindowMode mode, Consumer<String> listener) {
        String formatted = fitConversation(messages, maxTokens);
        return streamFormatted(formatted, maxTokens, temperature, topK, seed, penalty, mode, listener);
    }

    /** Streams the current prompt with recent visible chat turns. */
    public String streamConversation(String prompt, List<? extends ConversationTurn> history, int maxTokens,
                                     float temperature, int topK, long seed, float penalty, ContextWindowMode mode,
                                     Consumer<String> listener) {
        if (history.isEmpty()) return stream(prompt, maxTokens, temperature, topK, seed, penalty, mode, listener);
        var messages = new ArrayList<Qwen3Chat.Message>();
        for (var turn : history) {
            if (unfinishedThinking(turn.reply())) continue;
            messages.add(new Qwen3Chat.Message(Qwen3Chat.Role.USER, turn.prompt().replace("<|", "< |")));
            messages.add(new Qwen3Chat.Message(Qwen3Chat.Role.ASSISTANT, turn.reply().replace("<|", "< |")));
        }
        messages.add(new Qwen3Chat.Message(Qwen3Chat.Role.USER, prompt));
        return streamConversation(messages, maxTokens, temperature, topK, seed, penalty, mode, listener);
    }

    private static boolean unfinishedThinking(String text) {
        return text.stripLeading().startsWith("<think>") && !text.contains("</think>");
    }

    String fitConversation(List<Qwen3Chat.Message> messages, int maxTokens) {
        int window = config.maxSeqLen();
        int budget = window - Math.min(maxTokens, window / 2);
        int end = messages.size();
        if (end == 0 || messages.get(end - 1).role() != Qwen3Chat.Role.USER) {
            throw new IllegalArgumentException("A chat conversation must end with a user request");
        }
        String fitted = Qwen3Chat.format(messages.subList(end - 1, end), true, thinking);
        for (int index = end - 2; index >= 0; index--) {
            if (messages.get(index).role() != Qwen3Chat.Role.USER) continue;
            String candidate = Qwen3Chat.format(messages.subList(index, end), true, thinking);
            if (tokenizer.encode(candidate).length > budget) break;
            fitted = candidate;
        }
        return fitted;
    }

    private String streamFormatted(String formatted, int maxTokens, float temperature, int topK, long seed,
                                   float penalty, ContextWindowMode mode, Consumer<String> listener) {
        String prefix = tokenizer.decode(tokenizer.encode(formatted));
        try (var session = sessions.apply(mode)) {
            String text = TextGenerator.stream(session, config.maxSeqLen(), tokenizer, formatted,
                    maxTokens, temperature, topK, seed, penalty,
                    generated -> listener.accept(continuation(prefix, generated)));
            return continuation(prefix, text);
        }
    }

    private static String continuation(String prefix, String text) {
        return text.startsWith(prefix) ? text.substring(prefix.length()) : text;
    }
}
