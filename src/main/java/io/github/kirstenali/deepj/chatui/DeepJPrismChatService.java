package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.models.TextGenerator;

import java.io.IOException;
import java.nio.file.Path;
import java.util.function.Consumer;

public final class DeepJPrismChatService implements ChatService {

    private PrismCheckpoint.Loaded loaded;
    private Path checkpoint;

    @Override
    public void loadModel(Path modelPath) throws IOException {
        PrismCheckpoint.Loaded candidate = PrismCheckpoint.load(modelPath);
        loaded = candidate;
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
        validateRequest(prompt, maxTokens, temperature, topK);
        String prefix = loaded.tokenizer().decode(loaded.tokenizer().encode(prompt));
        String generated = TextGenerator.stream(loaded.model(), loaded.tokenizer(), loaded.config(),
                prompt, maxTokens, temperature, topK, seed,
                text -> listener.accept(GeneratedText.continuation(prefix, text)));
        return GeneratedText.continuation(prefix, generated);
    }

    private void validateRequest(String prompt, int maxTokens, float temperature, int topK) {
        if (loaded == null) throw new IllegalStateException("No model loaded");
        if (prompt == null || prompt.isBlank()) throw new IllegalArgumentException("Enter a message first");
        if (maxTokens <= 0) throw new IllegalArgumentException("Max tokens must be greater than zero");
        if (!Float.isFinite(temperature) || temperature <= 0.0f) {
            throw new IllegalArgumentException("Temperature must be greater than zero");
        }
        if (topK <= 0) throw new IllegalArgumentException("Top-k must be greater than zero");
    }
}
