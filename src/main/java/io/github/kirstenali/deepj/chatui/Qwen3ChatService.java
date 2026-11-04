package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.Agent;
import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3ChatModel;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tokenizers.qwen3.Qwen3Chat.Message;
import io.github.kirstenali.deepj.tokenizers.qwen3.Qwen3Chat.Role;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;

/** Adapts local Qwen3 snapshots to the existing desktop chat service contract. */
public final class Qwen3ChatService implements ConversationChatService {
    private Qwen3ChatModel chat;

    @Override public void loadModel(Path path) throws IOException { chat = loadSnapshot(path); }
    @Override public boolean isModelLoaded() { return chat != null; }
    @Override public String getLoadedModelName() { return chat == null ? "No model" : "Qwen3"; }

    static Qwen3ChatModel loadSnapshot(Path path) throws IOException {
        String dtype = System.getProperty("deepj.chat.qwen3.dtype");
        return Qwen3ChatModel.load(path, dtype == null ? null : DType.valueOf(dtype.toUpperCase(Locale.ROOT)));
    }

    @Override
    public Agent.Turn generateTurn(String prompt, List<Agent.Turn> history, ChatGenerationSettings settings,
                                   Consumer<String> listener) {
        if (chat == null) throw new IllegalStateException("Load a model first");
        return plainTurn(chat, prompt, history, settings, listener);
    }

    static Agent.Turn plainTurn(Qwen3ChatModel chat, String prompt, List<Agent.Turn> history,
                                ChatGenerationSettings settings, Consumer<String> listener) {
        String reply = chat.streamConversation(plainMessages(prompt, history), settings.maxTokens(),
                (float) settings.temperature(), settings.topK(), settings.seed(),
                (float) settings.repetitionPenalty(), settings.contextWindow(), listener);
        return Agent.Turn.answered(prompt, reply);
    }

    private static List<Message> plainMessages(String prompt, List<Agent.Turn> history) {
        var messages = new ArrayList<Message>();
        for (var turn : history) {
            messages.add(new Message(Role.USER, turn.prompt().replace("<|", "< |")));
            messages.add(new Message(Role.ASSISTANT, turn.reply().replace("<|", "< |"), ""));
        }
        messages.add(new Message(Role.USER, prompt));
        return messages;
    }

    @Override public String generate(String prompt, int maxTokens, float temperature, int topK, long seed,
                                     Consumer<String> listener) {
        return generate(prompt, maxTokens, temperature, topK, seed, 1, listener);
    }

    @Override public String generate(String prompt, int maxTokens, float temperature, int topK,
                                     long seed, float penalty, Consumer<String> listener) {
        return generate(prompt, maxTokens, temperature, topK, seed, penalty, ContextWindowMode.REBUILD, listener);
    }

    @Override public String generate(String prompt, int maxTokens, float temperature, int topK, long seed,
                                     float penalty, ContextWindowMode mode, Consumer<String> listener) {
        if (chat == null) throw new IllegalStateException("Load a model first");
        return chat.stream(prompt, maxTokens, temperature, topK, seed, penalty, mode, listener);
    }
}
