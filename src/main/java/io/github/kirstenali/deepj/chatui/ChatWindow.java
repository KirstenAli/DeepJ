package io.github.kirstenali.deepj.chatui;

import static io.github.kirstenali.deepj.chatui.DesktopErrors.message;
import io.github.kirstenali.deepj.agent.Agent;
import io.github.kirstenali.deepj.chatui.ChatRequests.Request;
import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;

import javafx.concurrent.Task;
import javafx.concurrent.Worker;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.web.WebView;
import javafx.stage.Stage;
import netscape.javascript.JSObject;

import java.io.File;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class ChatWindow {

    private final Stage stage;
    private final WebView view;
    private final ChatService service;
    private final AgentWorkspace agentWorkspace;
    private final ImageWorkspace imageWorkspace;
    private final VideoWorkspace videoWorkspace;
    private final ExecutorService worker = Executors.newSingleThreadExecutor(runnable -> {
        Thread thread = new Thread(runnable, "deepj-chat-worker");
        thread.setDaemon(true);
        return thread;
    });
    private Task<Agent.Turn> activeGeneration;
    private int activeResponse;
    private int activeAlternative = -1;
    private boolean busy;
    private int nextResponse;
    private final ChatRequests requests;

    ChatWindow(Stage stage, WebView view, ChatService service) {
        this.stage = stage;
        this.view = view;
        this.service = service;
        requests = new ChatRequests(service);
        imageWorkspace = new ImageWorkspace(stage, worker, service,
                new ImageWorkspace.Events(() -> busy, this::setBusy, requests::clear, this::call));
        videoWorkspace = new VideoWorkspace(stage, worker, service,
                new ImageWorkspace.Events(() -> busy, this::setBusy, requests::clear, this::call));
        agentWorkspace = service instanceof AgentChatService agent
                ? new AgentWorkspace(stage, agent, () -> busy) : null;
    }

    void open() {
        view.setContextMenuEnabled(false);
        stage.setOnHidden(event -> worker.shutdownNow());
        stage.getScene().addEventFilter(KeyEvent.KEY_PRESSED, this::handleEnter);
        view.getEngine().getLoadWorker().stateProperty().addListener((ob, old, state) -> {
            if (state == Worker.State.SUCCEEDED) connect();
        });
        view.getEngine().load(Objects.requireNonNull(getClass().getResource("/chat.html")).toExternalForm());
    }

    private void handleEnter(KeyEvent event) {
        if (event.getCode() != KeyCode.ENTER || event.isShiftDown() || !promptFocused()) return;
        call("submitPrompt");
        event.consume();
    }

    private boolean promptFocused() {
        if (view.getEngine().getLoadWorker().getState() != Worker.State.SUCCEEDED) return false;
        return Boolean.TRUE.equals(view.getEngine().executeScript(
                "document.activeElement === document.getElementById('prompt')"));
    }

    private void connect() {
        window().setMember("chatBridge", this);
        window().setMember("imageBridge", imageWorkspace);
        window().setMember("videoBridge", videoWorkspace);
        if (agentWorkspace != null) window().setMember("agentBridge", agentWorkspace);
        call("bridgeReady");
        refreshAgentWorkspace();
        call("setThinkingAvailable", service.supportsThinking());
    }

    public void chooseModel() {
        if (busy) return;
        File file = DesktopModelPicker.choose(stage);
        if (file != null) loadModel(file);
    }

    private void loadModel(File file) {
        if (DesktopModels.openMedia(file, imageWorkspace, videoWorkspace)) return;
        setBusy(true);
        call("setStatus", "Loading " + file.getName() + "…");
        DesktopModels.loadChat(file, service, worker, this::modelLoaded, this::loadFailed);
    }

    private void modelLoaded() {
        imageWorkspace.leave();
        videoWorkspace.leave();
        requests.clear();
        call("clearAlternativeActions");
        call("setModel", service.getLoadedModelName());
        call("setRollingCacheAvailable", service.supportsRollingCache());
        refreshAgentWorkspace();
        call("setThinkingAvailable", service.supportsThinking());
        call("setStatus", ChatBackend.readyStatus());
        setBusy(false);
    }

    private void refreshAgentWorkspace() {
        call("setAgentWorkspace", agentWorkspace == null ? "" : agentWorkspace.directoryLabel());
    }

    private void loadFailed(Throwable error) {
        call("setStatus", "Could not load model: " + message(error));
        setBusy(false);
    }

    public void send(String prompt, int maxTokens, double temperature, int topK, long seed) {
        send(prompt, maxTokens, temperature, topK, seed, 1.0);
    }

    // WebView needs an unambiguous method name when dispatching JavaScript calls.
    public void sendWithSettings(String prompt, int maxTokens, double temperature, int topK,
                                 long seed, double repetitionPenalty) {
        send(prompt, maxTokens, temperature, topK, seed, repetitionPenalty);
    }

    public void send(String prompt, int maxTokens, double temperature, int topK,
                     long seed, double repetitionPenalty) {
        sendWithWindowSettings(prompt, maxTokens, temperature, topK, seed, repetitionPenalty, false);
    }

    public void sendWithWindowSettings(String prompt, int maxTokens, double temperature, int topK,
                                       long seed, double repetitionPenalty, boolean rollingCache) {
        sendWithThinkingSettings(prompt, maxTokens, temperature, topK, seed, repetitionPenalty, rollingCache, false);
    }

    public void sendWithThinkingSettings(String prompt, int maxTokens, double temperature, int topK,
                                         long seed, double repetitionPenalty, boolean rollingCache, boolean thinking) {
        if (busy || !service.isModelLoaded()) return;
        if (prompt == null || prompt.isBlank()) return;
        try {
            startGeneration(prompt.trim(),
                    new ChatGenerationSettings(maxTokens, temperature, topK, seed, repetitionPenalty,
                            rollingCache ? ContextWindowMode.ROLLING : ContextWindowMode.REBUILD,
                            thinking && service.supportsThinking()));
        } catch (IllegalArgumentException error) {
            call("setStatus", error.getMessage());
        }
    }

    private void startGeneration(String prompt, ChatGenerationSettings settings) {
        int response = nextResponse++;
        var request = requests.add(response, prompt, settings);
        call("startResponse", prompt, response);
        call("setGenerating", true);
        setBusy(true);
        Task<Agent.Turn> task = requests.generate(request, settings);
        activeGeneration = task;
        activeResponse = response;
        task.messageProperty().addListener((ob, old, text) -> partial(task, response, text));
        task.setOnSucceeded(event -> complete(task, response));
        task.setOnFailed(event -> failed(task, response));
        worker.execute(task);
    }

    private void partial(Task<Agent.Turn> task, int response, String text) {
        if (task == activeGeneration && text != null && !text.isEmpty()) {
            call("updateResponse", response, text);
        }
    }

    private void complete(Task<Agent.Turn> task, int response) {
        if (task != activeGeneration) return;
        Agent.Turn turn = task.getValue();
        requests.complete(response, turn);
        call("finishResponse", response, turn.reply());
        finishGeneration();
    }

    private void failed(Task<Agent.Turn> task, int response) {
        if (task != activeGeneration) return;
        requests.remove(response);
        call("failResponse", response, message(task.getException()));
        finishGeneration();
    }

    public void alternatives(int response) {
        if (busy || !service.isModelLoaded()) return;
        Request request = requests.remove(response);
        if (request == null) return;
        setBusy(true);
        call("setGenerating", true);
        startAlternative(response, 0, request);
    }

    private void startAlternative(int response, int option, Request request) {
        ChatGenerationSettings settings = option == 0
                ? request.settings().withPenalty() : request.settings().withNewSeed();
        call("startAlternative", response, option);
        Task<Agent.Turn> task = requests.generate(request, settings);
        activeGeneration = task;
        activeResponse = response;
        activeAlternative = option;
        task.messageProperty().addListener((ob, old, text) -> partialAlternative(task, response, option, text));
        task.setOnSucceeded(event -> completeAlternative(task, response, option, request));
        task.setOnFailed(event -> failAlternative(task, response, option, request));
        worker.execute(task);
    }

    private void partialAlternative(Task<Agent.Turn> task, int response, int option, String text) {
        if (task == activeGeneration && text != null && !text.isEmpty()) {
            call("updateAlternative", response, option, text);
        }
    }

    private void completeAlternative(Task<Agent.Turn> task, int response, int option, Request request) {
        if (task != activeGeneration) return;
        call("finishAlternative", response, option, task.getValue().reply());
        nextAlternative(response, option, request);
    }

    private void failAlternative(Task<Agent.Turn> task, int response, int option, Request request) {
        if (task != activeGeneration) return;
        call("failAlternative", response, option, message(task.getException()));
        nextAlternative(response, option, request);
    }

    private void nextAlternative(int response, int option, Request request) {
        if (option == 0) startAlternative(response, 1, request);
        else finishGeneration();
    }

    public void stop() {
        if (activeGeneration == null) return;
        Task<Agent.Turn> task = activeGeneration;
        int response = activeResponse;
        activeGeneration = null;
        task.cancel(true);
        if (activeAlternative < 0) {
            requests.remove(response);
            call("stopResponse", response);
        } else {
            call("stopAlternative", response, activeAlternative);
        }
        finishGeneration();
    }

    public void newChat() {
        imageWorkspace.stop();
        imageWorkspace.clear();
        videoWorkspace.stop();
        videoWorkspace.clear();
        stop();
        requests.clear();
    }

    public void setConversationHistory(boolean enabled) {
        requests.useHistory(enabled);
    }

    private void finishGeneration() {
        activeGeneration = null;
        activeAlternative = -1;
        call("setGenerating", false);
        call("setStatus", ChatBackend.readyStatus());
        setBusy(false);
    }

    public void copy(String text) { DesktopClipboard.copy(text); }

    private void setBusy(boolean value) {
        busy = value;
        call("setBusy", value);
    }

    private JSObject window() {
        return (JSObject) view.getEngine().executeScript("window");
    }

    private void call(String name, Object... arguments) {
        window().call(name, arguments);
    }

}
