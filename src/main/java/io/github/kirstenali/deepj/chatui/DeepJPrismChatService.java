package io.github.kirstenali.deepj.chatui;

import java.io.IOException;
import java.nio.file.Path;
import java.util.function.Consumer;

public final class DeepJPrismChatService implements ChatService {

    private PrismCheckpoint.Loaded loaded;
    private Path checkpoint;

    @Override
    public void loadModel(Path modelPath) throws IOException {
        loaded = PrismCheckpoint.load(modelPath);
        checkpoint = modelPath;
    }

    @Override
    public boolean isModelLoaded() {
        return loaded != null;
    }

    @Override
    public String getLoadedModelName() {
        return checkpoint == null ? "No model" : checkpoint.getFileName().toString();
    }

    @Override
    public String generate(String prompt, int maxTokens, float temperature, int topK, long seed,
                           Consumer<String> listener) {
        if (loaded == null) throw new IllegalStateException("Load a model first");
        return ChatModelGeneration.stream(loaded.model(), loaded.tokenizer(), prompt,
                maxTokens, temperature, topK, seed, listener);
    }
}
