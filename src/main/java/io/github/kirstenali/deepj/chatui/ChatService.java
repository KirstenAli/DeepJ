package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import java.nio.file.Path;
import java.util.function.Consumer;

public interface ChatService {
    void loadModel(Path modelPath) throws Exception;

    boolean isModelLoaded();

    String getLoadedModelName();

    String generate(String prompt, int maxTokens, float temperature, int topK, long seed,
                    Consumer<String> listener);

    default String generate(String prompt, int maxTokens, float temperature, int topK,
                            long seed, float repetitionPenalty, Consumer<String> listener) {
        if (repetitionPenalty != 1.0f) {
            throw new UnsupportedOperationException("This chat service does not support repetition penalty");
        }
        return generate(prompt, maxTokens, temperature, topK, seed, listener);
    }

    default String generate(String prompt, int maxTokens, float temperature, int topK, long seed) {
        return generate(prompt, maxTokens, temperature, topK, seed, text -> {});
    }

    default String generate(String prompt, int maxTokens, float temperature, int topK, long seed,
                            float repetitionPenalty, ContextWindowMode mode, Consumer<String> listener) {
        if (java.util.Objects.requireNonNull(mode, "mode") == ContextWindowMode.ROLLING) {
            throw new UnsupportedOperationException("This chat service does not support rolling cached generation");
        }
        return generate(prompt, maxTokens, temperature, topK, seed, repetitionPenalty, listener);
    }
}
