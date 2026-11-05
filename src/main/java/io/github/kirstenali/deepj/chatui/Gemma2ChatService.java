package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.Agent;
import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.nn.transformer.gemma2.Gemma2ChatModel;
import io.github.kirstenali.deepj.tensor.DType;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;

/** Uses the same Gemma generation adapter as the CLI, with ordinary chat history and streaming. */
public final class Gemma2ChatService implements ConversationChatService {
    private Gemma2ChatModel chat;

    @Override public void loadModel(Path path) throws IOException {
        String selected = System.getProperty("deepj.chat.gemma2.dtype", "bfloat16");
        chat = Gemma2ChatModel.load(path, DType.valueOf(selected.toUpperCase(Locale.ROOT)));
    }

    @Override public void unloadModel() {
        if (chat != null) chat.model().close();
        chat = null;
    }
    @Override public boolean isModelLoaded() { return chat != null; }
    @Override public boolean supportsRollingCache() { return false; }
    @Override public String getLoadedModelName() { return chat == null ? "No model" : "Gemma 2"; }

    @Override public Agent.Turn generateTurn(String prompt, List<Agent.Turn> turns, ChatGenerationSettings settings,
                                            Consumer<String> listener) {
        if (chat == null) throw new IllegalStateException("Open a Gemma 2 model first");
        String reply = chat.stream(prompt, turns, settings.maxTokens(), (float) settings.temperature(),
                settings.topK(), settings.seed(), (float) settings.repetitionPenalty(),
                settings.contextWindow(), listener);
        return Agent.Turn.answered(prompt, reply);
    }

    @Override public String generate(String prompt, int count, float temperature, int topK, long seed,
                                     Consumer<String> listener) {
        return generate(prompt, count, temperature, topK, seed, 1, ContextWindowMode.REBUILD, listener);
    }

    @Override public String generate(String prompt, int count, float temperature, int topK, long seed,
                                     float penalty, Consumer<String> listener) {
        return generate(prompt, count, temperature, topK, seed, penalty, ContextWindowMode.REBUILD, listener);
    }

    @Override public String generate(String prompt, int count, float temperature, int topK, long seed,
                                     float penalty, ContextWindowMode mode, Consumer<String> listener) {
        return generateTurn(prompt, List.of(),
                new ChatGenerationSettings(count, temperature, topK, seed, penalty, mode), listener).reply();
    }
}
