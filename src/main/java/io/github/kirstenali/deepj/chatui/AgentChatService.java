package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.Agent;
import io.github.kirstenali.deepj.agent.InteractiveTerminal;
import io.github.kirstenali.deepj.agent.Qwen3AgentModel;
import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3ChatModel;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3Snapshot;

import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CancellationException;
import java.util.function.Consumer;
import java.util.function.Function;

final class AgentChatService implements ConversationChatService {
    private final Function<InteractiveTerminal.Request, Optional<InteractiveTerminal.Request>> review;
    private volatile Path directory = Path.of(".").toAbsolutePath().normalize();
    private volatile boolean enabled;
    private volatile Loaded model;

    AgentChatService(Function<InteractiveTerminal.Request, Optional<InteractiveTerminal.Request>> review) {
        this.review = review;
    }

    @Override
    public void loadModel(Path path) throws Exception {
        Loaded loaded = load(path);
        enabled = false;
        model = loaded;
    }

    private static Loaded load(Path path) throws Exception {
        if (Qwen3Snapshot.matches(path)) return new Loaded(null, Qwen3ChatService.loadSnapshot(path));
        ChatService plain = new DeepJChatService();
        plain.loadModel(path);
        return new Loaded(plain, null);
    }

    void enable(boolean selected) { enabled = selected; }

    boolean supportsAgent() {
        Loaded loaded = model;
        return loaded != null && loaded.qwen() != null && InteractiveTerminal.available();
    }

    void directory(Path path) { directory = path; }

    Path directory() { return directory; }

    @Override public void unloadModel() {
        model = null;
        enabled = false;
    }

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
        var settings = new ChatGenerationSettings(maxTokens, temperature, topK, seed, penalty, mode);
        return generateTurn(prompt, List.of(), settings, listener).reply();
    }

    @Override
    public Agent.Turn generateTurn(String prompt, List<Agent.Turn> history, ChatGenerationSettings settings,
                                   Consumer<String> listener) {
        Loaded loaded = model;
        if (loaded == null) throw new IllegalStateException("Load a model first");
        if (loaded.qwen() == null) {
            return settings.generateTurn(loaded.plain(), prompt, history, listener);
        }
        if (!enabled) {
            return Qwen3ChatService.plainTurn(loaded.qwen(), prompt, history, settings, listener);
        }
        var qwen = new Qwen3AgentModel(loaded.qwen(), settings.maxTokens(), (float) settings.temperature(),
                settings.topK(), settings.seed(), (float) settings.repetitionPenalty(), settings.contextWindow());
        return runAgent(prompt, history, qwen, listener);
    }

    private Agent.Turn runAgent(String prompt, List<Agent.Turn> history, Qwen3AgentModel qwen,
                                Consumer<String> listener) {
        try {
            var agent = InteractiveTerminal.withCommandReview(directory, review).agent(qwen);
            return agent.runTurn(history, prompt, listener);
        } catch (InterruptedException stopped) {
            Thread.currentThread().interrupt();
            throw new CancellationException("Agent stopped");
        }
    }

    private record Loaded(ChatService plain, Qwen3ChatModel qwen) {}
}
