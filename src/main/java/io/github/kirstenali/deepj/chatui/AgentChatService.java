package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.InteractiveTerminal;
import io.github.kirstenali.deepj.agent.Qwen3AgentModel;
import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3ChatModel;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3Snapshot;

import java.nio.file.Path;
import java.util.concurrent.CancellationException;
import java.util.function.Consumer;
import java.util.function.Predicate;

final class AgentChatService implements ChatService {
    private final Consumer<Boolean> modelChanged;
    private final Predicate<InteractiveTerminal.Request> approve;
    private volatile Path directory = Path.of(".").toAbsolutePath().normalize();
    private volatile boolean enabled;
    private volatile Loaded model;

    AgentChatService(Consumer<Boolean> modelChanged, Predicate<InteractiveTerminal.Request> approve) {
        this.modelChanged = modelChanged;
        this.approve = approve;
    }

    @Override
    public void loadModel(Path path) throws Exception {
        Loaded loaded = load(path);
        enabled = false;
        model = loaded;
        modelChanged.accept(loaded.qwen() != null && InteractiveTerminal.available());
    }

    private static Loaded load(Path path) throws Exception {
        if (Qwen3Snapshot.matches(path)) return new Loaded(null, Qwen3ChatModel.load(path));
        ChatService plain = new DeepJChatService();
        plain.loadModel(path);
        return new Loaded(plain, null);
    }

    void enable(boolean selected) { enabled = selected; }

    void directory(Path path) { directory = path; }

    Path directory() { return directory; }

    @Override
    public boolean isModelLoaded() { return model != null; }

    @Override
    public String getLoadedModelName() {
        Loaded loaded = model;
        if (loaded == null) return "No model";
        return loaded.qwen() == null ? loaded.plain().getLoadedModelName() : "Qwen3";
    }

    @Override
    public String generate(String prompt, int maxTokens, float temperature, int topK, long seed,
                           Consumer<String> listener) {
        return generate(prompt, maxTokens, temperature, topK, seed, 1, ContextWindowMode.REBUILD, listener);
    }

    @Override
    public String generate(String prompt, int maxTokens, float temperature, int topK, long seed,
                           float penalty, ContextWindowMode mode, Consumer<String> listener) {
        Loaded loaded = model;
        if (loaded == null) throw new IllegalStateException("Load a model first");
        if (loaded.qwen() == null) {
            return loaded.plain().generate(prompt, maxTokens, temperature, topK, seed, penalty, mode, listener);
        }
        if (!enabled) return loaded.qwen().stream(prompt, maxTokens, temperature, topK, seed, penalty, mode, listener);
        return runAgent(prompt,
                new Qwen3AgentModel(loaded.qwen(), maxTokens, temperature, topK, seed, penalty, mode));
    }

    private String runAgent(String prompt, Qwen3AgentModel qwen) {
        try {
            return new InteractiveTerminal(directory, approve).agent(qwen).run(prompt);
        } catch (InterruptedException stopped) {
            Thread.currentThread().interrupt();
            throw new CancellationException("Agent stopped");
        }
    }

    private record Loaded(ChatService plain, Qwen3ChatModel qwen) {}
}
