package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.models.origin.DeepJOrigin;
import io.github.kirstenali.deepj.models.TextGenerator;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.io.IOException;
import java.nio.file.Path;

public final class DeepJOriginChatService implements ChatService {

    private final Tokenizer tokenizer;
    private final DeepJOriginConfig config;

    private DeepJOrigin loadedModel;
    private Path loadedModelPath;

    public DeepJOriginChatService() {
        this.tokenizer = new ByteTokenizer();
        this.config = new DeepJOriginConfig(
                ByteTokenizer.VOCAB_SIZE,
                128,
                256,
                4,
                5,
                1024
        );
    }

    @Override
    public void loadModel(Path modelPath) throws IOException {
        DeepJOrigin model = new DeepJOrigin(config, 42);
        model.load(modelPath);

        this.loadedModel = model;
        this.loadedModelPath = modelPath;
    }

    @Override
    public boolean isModelLoaded() {
        return loadedModel != null;
    }

    @Override
    public String getLoadedModelName() {
        return loadedModelPath == null ? "None" : loadedModelPath.getFileName().toString();
    }

    @Override
    public String generate(String prompt, int maxTokens, float temperature, int topK, long seed) {
        validateRequest(prompt, maxTokens, temperature, topK);
        return TextGenerator.generate(loadedModel, tokenizer, config, prompt,
                maxTokens, temperature, topK, seed);
    }

    private void validateRequest(String prompt, int maxTokens, float temperature, int topK) {
        if (loadedModel == null) throw new IllegalStateException("No model loaded.");
        if (prompt == null || prompt.isBlank()) throw new IllegalArgumentException("Prompt must not be empty.");
        if (maxTokens <= 0) throw new IllegalArgumentException("Max tokens must be greater than 0.");
        if (!Float.isFinite(temperature) || temperature <= 0.0f) {
            throw new IllegalArgumentException("Temperature must be finite and greater than 0.");
        }
        if (topK <= 0) throw new IllegalArgumentException("Top-k must be greater than 0.");
    }
}
