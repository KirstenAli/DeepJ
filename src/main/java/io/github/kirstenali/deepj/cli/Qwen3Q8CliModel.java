package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.agent.Agent;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3ChatSupport;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3Gguf;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3Config;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3Q8;
import io.github.kirstenali.deepj.persistence.ModelInspection;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tokenizers.qwen3.Qwen3Tokenizer;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Consumer;

record Qwen3Q8CliModel(Qwen3Q8 model, Qwen3Tokenizer tokenizer, ModelInspection inspection,
                       TensorBackend computation) implements CliModel {
    static CliModel load(Path path, BackendChoice choice) throws IOException {
        ModelInspection inspection = Qwen3Gguf.inspect(path);
        var tokenizer = Qwen3Tokenizer.load(path.toAbsolutePath().getParent());
        var config = (Qwen3Config) inspection.settings().config();
        if (tokenizer.vocabSize() > config.vocabSize()) throw new IOException("Tokenizer exceeds GGUF vocabulary");
        choice.activate();
        return new Qwen3Q8CliModel(Qwen3Gguf.load(path), tokenizer, inspection, Tensor.backend());
    }

    @Override public String name() { return "Qwen3 Q8_0"; }
    @Override public boolean supportsThinking() { return true; }
    @Override public String backend() { return computation.usesDeviceStorage() ? "Metal" : "CPU"; }
    @Override public void close() { model.close(); }

    @Override public String stream(String prompt, SamplingSettings settings, Consumer<String> listener) {
        return streamConversation(prompt, List.of(), settings, listener);
    }

    @Override public String streamConversation(String prompt, List<Agent.Turn> history, SamplingSettings settings,
                                               Consumer<String> listener) {
        var chat = new Qwen3ChatSupport(model.config(), tokenizer, settings.thinking(), model::openGeneration);
        return chat.streamConversation(prompt, history, settings.maxTokens(), settings.temperature(), settings.topK(),
                settings.seed(), settings.repetitionPenalty(), settings.contextWindow(), listener);
    }
}
