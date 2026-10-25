package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.models.ContextWindowMode;

import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.models.ModelLoader;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.Consumer;

final class EmbeddedModelChatService implements ChatService {

    @Override
    public String generate(String prompt, int maxTokens, float temperature, int topK, long seed,
                           float repetitionPenalty, ContextWindowMode mode, Consumer<String> listener) {
        if (model == null) throw new IllegalStateException("Load a model first");
        return ChatModelGeneration.stream(model, tokenizer, prompt, maxTokens, temperature, topK,
                seed, repetitionPenalty, mode, listener);
    }

    private DecoderOnlyModel model;
    private Tokenizer tokenizer;
    private Path checkpoint;

    @Override
    public void loadModel(Path path) throws IOException {
        DecoderOnlyModel loaded = ModelLoader.load(path);
        Tokenizer selected = tokenizer(path);
        if (selected.vocabSize() != loaded.config().vocabSize()) {
            throw new IOException("Tokenizer vocabulary does not match model");
        }
        model = loaded;
        tokenizer = selected;
        checkpoint = path;
    }

    private static Tokenizer tokenizer(Path path) throws IOException {
        Path bpe = path.toAbsolutePath().getParent().resolve("tokenizer.bpe");
        return Files.isRegularFile(bpe) ? new BPETokenizer(BPEModelIO.load(bpe)) : new ByteTokenizer();
    }

    @Override
    public boolean isModelLoaded() {
        return model != null;
    }

    @Override
    public String getLoadedModelName() {
        return checkpoint == null ? "No model" : checkpoint.getFileName().toString();
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
        return ChatModelGeneration.stream(model, tokenizer, prompt,
                maxTokens, temperature, topK, seed, repetitionPenalty, listener);
    }
}
