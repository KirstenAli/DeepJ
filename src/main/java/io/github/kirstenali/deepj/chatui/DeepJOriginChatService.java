package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.Agent;
import io.github.kirstenali.deepj.nn.transformer.PromptFormat;
import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOrigin;
import io.github.kirstenali.deepj.nn.transformer.TextGenerator;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Consumer;

public final class DeepJOriginChatService implements ConversationChatService {

    @Override
    public Agent.Turn generateTurn(String prompt, List<Agent.Turn> history, ChatGenerationSettings settings,
                                   Consumer<String> listener) {
        String formatted = PromptFormat.PLAIN.applyConversation(prompt, history, tokenizer,
                config.maxSeqLen(), settings.maxTokens());
        return Agent.Turn.answered(prompt, settings.generate(this, formatted, listener));
    }

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
    public String generate(String prompt, int maxTokens, float temperature, int topK, long seed,
                           Consumer<String> listener) {
        return generate(prompt, maxTokens, temperature, topK, seed, 1.0f, listener);
    }

    @Override
    public String generate(String prompt, int maxTokens, float temperature, int topK,
                           long seed, float repetitionPenalty, Consumer<String> listener) {
        validateRequest(prompt, maxTokens, temperature, topK);
        String prefix = tokenizer.decode(tokenizer.encode(prompt));
        String result = TextGenerator.stream(loadedModel, tokenizer, config, prompt,
                maxTokens, temperature, topK, seed, repetitionPenalty,
                text -> listener.accept(GeneratedText.continuation(prefix, text)));
        return GeneratedText.continuation(prefix, result);
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
