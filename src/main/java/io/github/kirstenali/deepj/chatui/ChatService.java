package io.github.kirstenali.deepj.chatui;

import java.nio.file.Path;
import java.util.function.Consumer;

public interface ChatService {
    void loadModel(Path modelPath) throws Exception;

    boolean isModelLoaded();

    String getLoadedModelName();

    String generate(String prompt, int maxTokens, float temperature, int topK, long seed,
                    Consumer<String> listener);

    default String generate(String prompt, int maxTokens, float temperature, int topK, long seed) {
        return generate(prompt, maxTokens, temperature, topK, seed, text -> {});
    }
}
