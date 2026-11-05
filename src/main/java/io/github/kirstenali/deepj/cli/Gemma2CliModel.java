package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.agent.Agent;
import io.github.kirstenali.deepj.nn.transformer.gemma2.Gemma2ChatModel;
import io.github.kirstenali.deepj.nn.transformer.gemma2.Gemma2Pretrained;
import io.github.kirstenali.deepj.nn.transformer.gemma2.Gemma2Snapshot;
import io.github.kirstenali.deepj.persistence.ModelInspection;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Consumer;

record Gemma2CliModel(Gemma2ChatModel chat, ModelInspection inspection, TensorBackend computation) implements CliModel {
    static CliModel load(Path selection, BackendChoice backend, DType dtype) throws IOException {
        Path folder = Gemma2Snapshot.directory(selection);
        var details = Gemma2Pretrained.inspect(folder);
        backend.activate();
        try {
            return new Gemma2CliModel(Gemma2ChatModel.load(folder, dtype), details, Tensor.backend());
        } catch (IOException | RuntimeException | Error error) {
            Tensor.backend().releaseResources();
            throw error;
        }
    }

    @Override public String name() { return "Gemma 2"; }
    @Override public String backend() { return computation instanceof MetalBackend ? "Metal" : "CPU"; }
    @Override public void close() {
        chat.model().close();
        computation.releaseResources();
    }

    @Override public String stream(String prompt, SamplingSettings settings, Consumer<String> listener) {
        return streamConversation(prompt, List.of(), settings, listener);
    }

    @Override public String streamConversation(String prompt, List<Agent.Turn> turns, SamplingSettings settings,
                                               Consumer<String> listener) {
        return chat.stream(prompt, turns, settings.maxTokens(), settings.temperature(), settings.topK(),
                settings.seed(), settings.repetitionPenalty(), settings.contextWindow(), listener);
    }
}
