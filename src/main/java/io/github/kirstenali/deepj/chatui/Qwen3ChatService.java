package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3ChatModel;
import java.io.IOException;
import java.nio.file.Path;
import java.util.function.Consumer;

/** Adapts local Qwen3 snapshots to the existing desktop chat service contract. */
public final class Qwen3ChatService implements ChatService {
    private Qwen3ChatModel chat;

    @Override public void loadModel(Path path) throws IOException { chat = Qwen3ChatModel.load(path); }
    @Override public boolean isModelLoaded() { return chat != null; }
    @Override public String getLoadedModelName() { return chat == null ? "No model" : "Qwen3"; }

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
