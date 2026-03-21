package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.models.gpt.GPTConfig;
import io.github.kirstenali.deepj.models.gpt.GPTModel;
import io.github.kirstenali.deepj.models.gpt.TextGenerator;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.io.IOException;
import java.nio.file.Path;

public final class GPTChatService implements ChatService {

    private final Tokenizer tokenizer;
    private final GPTConfig config;

    private GPTModel loadedModel;
    private Path loadedModelPath;

    public GPTChatService() {
        this.tokenizer = new ByteTokenizer();
        this.config = new GPTConfig(
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
        GPTModel model = new GPTModel(config, 42);
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
    public String generate(String prompt, int maxTokens, double temperature, int topK, long seed) {
        if (loadedModel == null) {
            throw new IllegalStateException("No model loaded.");
        }

        if (prompt == null || prompt.isBlank()) {
            throw new IllegalArgumentException("Prompt must not be empty.");
        }

        if (maxTokens <= 0) {
            throw new IllegalArgumentException("Max tokens must be greater than 0.");
        }

        if (temperature < 0.0) {
            throw new IllegalArgumentException("Temperature must be >= 0.");
        }

        if (topK <= 0) {
            throw new IllegalArgumentException("Top-k must be greater than 0.");
        }

        return TextGenerator.generate(
                loadedModel,
                tokenizer,
                config,
                prompt,
                maxTokens,
                temperature,
                topK,
                seed
        );
    }
}