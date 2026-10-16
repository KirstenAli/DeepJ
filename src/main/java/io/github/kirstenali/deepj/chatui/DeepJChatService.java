package io.github.kirstenali.deepj.chatui;

import java.nio.file.Path;
import java.util.function.Consumer;

public final class DeepJChatService implements ChatService {

    private ChatService model;

    @Override
    public void loadModel(Path path) throws Exception {
        ChatService candidate = serviceFor(path);
        candidate.loadModel(path);
        model = candidate;
    }

    private static ChatService serviceFor(Path path) {
        return path.toString().endsWith(".dj")
                ? new EmbeddedModelChatService() : new DeepJOriginChatService();
    }

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
        if (model == null) throw new IllegalStateException("Load a model first");
        return model.generate(prompt, maxTokens, temperature, topK, seed, listener);
    }
}
