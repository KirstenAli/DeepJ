package io.github.kirstenali.deepj.agent;

import io.github.kirstenali.deepj.nn.transformer.TextGenerator;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3ChatModel;
import io.github.kirstenali.deepj.tokenizers.qwen3.Qwen3Chat;

import java.util.List;
import java.util.Objects;

/** Adapts DeepJ's Qwen3 model and tokenizer to the agent's conversation format. */
public record Qwen3AgentModel(Qwen3ChatModel chat, int maxTokens, float temperature, int topK, long seed)
        implements AgentModel {

    public Qwen3AgentModel(Qwen3ChatModel chat) {
        this(chat, 256, 0.7f, 20, 1234L);
    }

    public Qwen3AgentModel {
        Objects.requireNonNull(chat, "chat");
        if (maxTokens < 1) throw new IllegalArgumentException("maxTokens must be positive");
        if (!Float.isFinite(temperature) || temperature <= 0) throw new IllegalArgumentException("Invalid temperature");
        if (topK < 0) throw new IllegalArgumentException("topK must be nonnegative");
    }

    @Override
    public String generate(List<Message> messages) {
        String prompt = Qwen3Chat.format(messages.stream().map(Qwen3AgentModel::format).toList(), true, false);
        var tokenizer = chat.tokenizer();
        int window = chat.model().config().maxSeqLen();
        int[] input = tokenizer.encode(prompt);
        int tokens = Math.min(maxTokens, window - input.length);
        if (tokens < 1) throw new IllegalStateException("Agent conversation fills the model context window");
        String prefix = tokenizer.decode(input);
        try (var session = chat.model().openGeneration()) {
            String text = TextGenerator.generate(session, window, tokenizer, prompt, tokens, temperature, topK, seed);
            if (!text.startsWith(prefix)) throw new IllegalStateException("Generated text lost its prompt prefix");
            return text.substring(prefix.length());
        }
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
