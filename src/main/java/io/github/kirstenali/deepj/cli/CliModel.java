package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.nn.transformer.DeepJChat;
import io.github.kirstenali.deepj.persistence.ModelInspection;
import io.github.kirstenali.deepj.persistence.ModelSerializer;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.Consumer;

interface CliModel extends AutoCloseable {
    String name();
    String backend();
    ModelInspection inspection();
    String stream(String prompt, SamplingSettings settings, Consumer<String> listener);
    @Override void close();

    static Path modelFile(Path path) {
        return Files.isDirectory(path) ? path.resolve("model.dj") : path;
    }

    static String architecture(ModelInspection info) {
        return info.settings().config().getClass().getSimpleName().replace("Config", "").replace("DeepJ", "DeepJ ");
    }

    static CliModel load(Path path, BackendChoice backend) throws IOException {
        Path file = modelFile(path).toAbsolutePath();
        ModelInspection info = ModelSerializer.inspect(file);
        backend.activate();
        try {
            String name = info.settings().metadata().getOrDefault("name", architecture(info));
            return new Loaded(DeepJChat.load(file), info, name, Tensor.backend());
        } catch (IOException | RuntimeException | Error failure) {
            Tensor.backend().releaseResources();
            throw failure;
        }
    }

    record Loaded(DeepJChat chat, ModelInspection inspection, String name,
                  TensorBackend computation) implements CliModel {
        @Override
        public String backend() { return computation instanceof MetalBackend ? "Metal" : "CPU"; }

        @Override
        public String stream(String prompt, SamplingSettings settings, Consumer<String> listener) {
            return chat.withContextWindow(settings.contextWindow()).stream(
                    prompt, settings.maxTokens(), settings.temperature(), settings.topK(),
                    settings.seed(), settings.repetitionPenalty(), listener);
        }

        @Override
        public void close() { computation.releaseResources(); }
    }
}
