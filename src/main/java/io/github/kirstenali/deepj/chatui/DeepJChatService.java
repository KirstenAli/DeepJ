package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.Agent;
import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3Snapshot;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Consumer;

public final class DeepJChatService implements ConversationChatService {

    @Override
    public Agent.Turn generateTurn(String prompt, List<Agent.Turn> history, ChatGenerationSettings settings,
                                   Consumer<String> listener) {
        if (model == null) throw new IllegalStateException("Load a model first");
        return settings.generateTurn(model, prompt, history, listener);
    }

    @Override
    public String generate(String prompt, int maxTokens, float temperature, int topK, long seed,
                           float repetitionPenalty, ContextWindowMode mode, Consumer<String> listener) {
        if (model == null) throw new IllegalStateException("Load a model first");
        return model.generate(prompt, maxTokens, temperature, topK, seed, repetitionPenalty, mode, listener);
    }

    private ChatService model;

    @Override
    public void loadModel(Path path) throws Exception {
        ChatService candidate = serviceFor(path);
        candidate.loadModel(path);
        model = candidate;
    }

    private static ChatService serviceFor(Path path) throws IOException {
        if (Qwen3Snapshot.matches(path)) return new Qwen3ChatService();
        return path.toString().endsWith(".dj")
                ? new EmbeddedModelChatService() : new DeepJOriginChatService();
    }

    @Override public void unloadModel() { model = null; }

    @Override
    public boolean isModelLoaded() {
        return model != null && model.isModelLoaded();
    }

    @Override
    public String getLoadedModelName() {
        return model == null ? "No model" : model.getLoadedModelName();
    }

    @Override
    public String generate(String prompt, int maxTokens, float temperature, int topK, long seed,
                           Consumer<String> listener) {
        return generate(prompt, maxTokens, temperature, topK, seed, 1.0f, listener);
    }

    @Override
    public String generate(String prompt, int maxTokens, float temperature, int topK,
                           long seed, float repetitionPenalty, Consumer<String> listener) {
        if (model == null) throw new IllegalStateException("Load a model first");
        return model.generate(prompt, maxTokens, temperature, topK, seed, repetitionPenalty, listener);
    }
}
