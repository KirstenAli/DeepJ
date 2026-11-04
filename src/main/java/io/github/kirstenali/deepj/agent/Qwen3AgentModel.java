package io.github.kirstenali.deepj.agent;

import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.nn.transformer.TextGenerator;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3ChatModel;
import io.github.kirstenali.deepj.tokenizers.qwen3.Qwen3Chat;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;

/** Adapts DeepJ's Qwen3 model and tokenizer to the agent's conversation format. */
public record Qwen3AgentModel(Qwen3ChatModel chat, int maxTokens, float temperature, int topK,
                              long seed, float repetitionPenalty, ContextWindowMode mode)
        implements AgentModel {

    public Qwen3AgentModel(Qwen3ChatModel chat) {
        this(chat, 256, 0.7f, 20, 1234L);
    }

    public Qwen3AgentModel(Qwen3ChatModel chat, int maxTokens, float temperature, int topK, long seed) {
        this(chat, maxTokens, temperature, topK, seed, 1.0f, ContextWindowMode.REBUILD);
    }

    public Qwen3AgentModel {
        Objects.requireNonNull(chat, "chat");
        Objects.requireNonNull(mode, "mode");
        if (maxTokens < 1) throw new IllegalArgumentException("maxTokens must be positive");
        if (!Float.isFinite(temperature) || temperature <= 0) throw new IllegalArgumentException("Invalid temperature");
        if (topK < 0) throw new IllegalArgumentException("topK must be nonnegative");
        if (!Float.isFinite(repetitionPenalty) || repetitionPenalty < 1) {
            throw new IllegalArgumentException("Repetition penalty must be at least 1");
        }
    }

    @Override
    public String generate(List<Message> messages) {
        return stream(messages, null);
    }

    @Override
    public String stream(List<Message> messages, Consumer<String> listener) {
        int window = chat.model().config().maxSeqLen();
        Prompt input = fitPrompt(messages, window);
        int tokens = Math.min(maxTokens, window - input.ids().length);
        if (tokens < 1) throw new IllegalStateException("Agent conversation fills the model context window");
        return generate(input, window, tokens, listener);
    }

    private String generate(Prompt input, int window, int tokens, Consumer<String> listener) {
        var tokenizer = chat.tokenizer();
        String prefix = tokenizer.decode(input.ids());
        try (var session = chat.model().openGeneration(mode)) {
            String text = listener == null
                    ? TextGenerator.generate(session, window, tokenizer, input.text(),
                            tokens, temperature, topK, seed, repetitionPenalty)
                    : TextGenerator.stream(session, window, tokenizer, input.text(),
                            tokens, temperature, topK, seed, repetitionPenalty,
                            generated -> listener.accept(continuation(prefix, generated)));
            return continuation(prefix, text);
        }
    }

    private Prompt fitPrompt(List<Message> messages, int window) {
        Prompt prompt = encode(messages);
        int budget = window - Math.min(maxTokens, window - 1);
        if (prompt.ids().length <= budget) return prompt;
        int firstUser = messages.get(0).role() == Role.SYSTEM ? 1 : 0;
        int current = messages.size() - 1;
        while (current > firstUser && messages.get(current).role() != Role.USER) {
            current--;
        }
        for (int start = firstUser + 1; start <= current; start++) {
            if (messages.get(start).role() != Role.USER) continue;
            var recent = new ArrayList<>(messages.subList(0, firstUser));
            recent.addAll(messages.subList(start, messages.size()));
            prompt = encode(recent);
            if (prompt.ids().length <= budget) break;
        }
        return prompt;
    }

    private Prompt encode(List<Message> messages) {
        String prompt = Qwen3Chat.format(messages.stream().map(Qwen3AgentModel::format).toList(), true, false);
        return new Prompt(prompt, chat.tokenizer().encode(prompt));
    }

    private record Prompt(String text, int[] ids) {}

    private static String continuation(String prefix, String text) {
        if (!text.startsWith(prefix)) throw new IllegalStateException("Generated text lost its prompt prefix");
        return text.substring(prefix.length());
    }

    private static Qwen3Chat.Message format(Message message) {
        Qwen3Chat.Role role = switch (message.role()) {
            case SYSTEM -> Qwen3Chat.Role.SYSTEM;
            case USER, TOOL -> Qwen3Chat.Role.USER;
            case ASSISTANT -> Qwen3Chat.Role.ASSISTANT;
        };
        String content = message.content().replace("<|", "< |");
        return new Qwen3Chat.Message(role, content, role == Qwen3Chat.Role.ASSISTANT ? "" : null);
    }
}
