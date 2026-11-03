package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3ChatModel;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3Pretrained;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3Snapshot;
import io.github.kirstenali.deepj.persistence.ModelInspection;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.io.IOException;
import java.nio.file.Path;
import java.util.function.Consumer;

record Qwen3CliModel(Qwen3ChatModel chat, ModelInspection inspection, TensorBackend computation) implements CliModel {
    static CliModel load(Path selection, BackendChoice choice, DType dtype) throws IOException {
        Path directory = Qwen3Snapshot.directory(selection);
        ModelInspection inspection = Qwen3Pretrained.inspect(directory);
        choice.activate();
        try { return new Qwen3CliModel(Qwen3ChatModel.load(directory, dtype), inspection, Tensor.backend()); }
        catch (IOException | RuntimeException | Error failure) {
            Tensor.backend().releaseResources();
            throw failure;
        }
    }

    @Override public String name() { return "Qwen3"; }
    @Override public String backend() { return computation instanceof MetalBackend ? "Metal" : "CPU"; }
    @Override public void close() { computation.releaseResources(); }

    @Override public String stream(String prompt, SamplingSettings settings, Consumer<String> listener) {
        return chat.stream(prompt, settings.maxTokens(), settings.temperature(), settings.topK(), settings.seed(),
                settings.repetitionPenalty(), settings.contextWindow(), listener);
    }
}
