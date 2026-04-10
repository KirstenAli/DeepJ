package io.github.kirstenali.deepj.chatui;

import java.nio.file.Path;

public interface ChatService {
    void loadModel(Path modelPath) throws Exception;

    boolean isModelLoaded();

    String getLoadedModelName();

    String generate(String prompt, int maxTokens, float temperature, int topK, long seed);
}
