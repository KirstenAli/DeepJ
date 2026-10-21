package io.github.kirstenali.deepj.chatui;

import javafx.concurrent.Task;
import javafx.concurrent.Worker;
import javafx.scene.input.Clipboard;
import javafx.scene.input.ClipboardContent;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.web.WebView;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import netscape.javascript.JSObject;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class ChatWindow {

    private final Stage stage;
    private final WebView view;
    private final ChatService service;
    private final ExecutorService worker = Executors.newSingleThreadExecutor(runnable -> {
        Thread thread = new Thread(runnable, "deepj-chat-worker");
        thread.setDaemon(true);
        return thread;
    });
    private Task<String> activeGeneration;
    private int activeResponse;
    private int activeAlternative = -1;
    private boolean busy;
    private int nextResponse;
    private final Map<Integer, Request> requests = new HashMap<>();

    ChatWindow(Stage stage, WebView view, ChatService service) {
        this.stage = stage;
        this.view = view;
        this.service = service;
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
        call("bridgeReady");
    }

    public void chooseModel() {
        if (busy) return;
        File file = modelFile();
        if (file != null) loadModel(file);
    }

    private File modelFile() {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Choose a DeepJ checkpoint");
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("DeepJ models", "*.dj", "*.bin"));
        return chooser.showOpenDialog(stage);
    }

    private void loadModel(File file) {
        setBusy(true);
        call("setStatus", "Loading " + file.getName() + "…");
        Task<Void> task = new Task<>() {
            @Override protected Void call() throws Exception {
                service.loadModel(file.toPath());
                return null;
            }
        };
        task.setOnSucceeded(event -> modelLoaded());
        task.setOnFailed(event -> loadFailed(task.getException()));
        start(task);
    }

    private void modelLoaded() {
        requests.clear();
        call("clearAlternativeActions");
        call("setModel", service.getLoadedModelName());
        call("setStatus", ChatBackend.readyStatus());
        setBusy(false);
    }

    private void loadFailed(Throwable error) {
        call("setStatus", "Could not load model: " + message(error));
        setBusy(false);
    }

    public void send(String prompt, int maxTokens, double temperature, int topK, long seed) {
        send(prompt, maxTokens, temperature, topK, seed, 1.0);
    }

    public void send(String prompt, int maxTokens, double temperature, int topK,
                     long seed, double repetitionPenalty) {
        System.out.println("send6: prompt=" + prompt + " busy=" + busy + " loaded=" + service.isModelLoaded()
                + " penalty=" + repetitionPenalty + " service=" + service.getClass().getName());
        if (busy || !service.isModelLoaded()) return;
        if (prompt == null || prompt.isBlank()) return;
        try {
            startGeneration(prompt.trim(),
                    new ChatGenerationSettings(maxTokens, temperature, topK, seed, repetitionPenalty));
        } catch (IllegalArgumentException error) {
            call("setStatus", error.getMessage());
        }
    }

    private void startGeneration(String prompt, ChatGenerationSettings settings) {
        int response = nextResponse++;
        requests.put(response, new Request(prompt, settings));
        call("startResponse", prompt, response);
        call("setGenerating", true);
        setBusy(true);
        Task<String> task = generation(prompt, settings);
        activeGeneration = task;
        activeResponse = response;
        task.valueProperty().addListener((ob, old, text) -> partial(task, response, text));
        task.setOnSucceeded(event -> complete(task, response));
        task.setOnFailed(event -> failed(task, response));
        start(task);
    }

    private Task<String> generation(String prompt, ChatGenerationSettings settings) {
        return new Task<>() {
            @Override protected String call() {
                return service.generate(prompt, settings.maxTokens(), (float) settings.temperature(),
                        settings.topK(), settings.seed(), (float) settings.repetitionPenalty(), this::updateValue);
            }
        };
    }

    private void partial(Task<String> task, int response, String text) {
        if (task == activeGeneration && text != null && !text.isEmpty()) {
            call("updateResponse", response, text);
        }
    }

    private void complete(Task<String> task, int response) {
        if (task != activeGeneration) return;
        if (task.getValue() == null || task.getValue().isBlank()) requests.remove(response);
        call("finishResponse", response, task.getValue());
        finishGeneration();
    }

    private void failed(Task<String> task, int response) {
        if (task != activeGeneration) return;
        requests.remove(response);
        call("failResponse", response, "Generation failed: " + message(task.getException()));
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
        Task<String> task = generation(request.prompt(), settings);
        activeGeneration = task;
        activeResponse = response;
        activeAlternative = option;
        task.valueProperty().addListener((ob, old, text) -> partialAlternative(task, response, option, text));
        task.setOnSucceeded(event -> completeAlternative(task, response, option, request));
        task.setOnFailed(event -> failAlternative(task, response, option, request));
        start(task);
    }

    private void partialAlternative(Task<String> task, int response, int option, String text) {
        if (task == activeGeneration && text != null && !text.isEmpty()) {
            call("updateAlternative", response, option, text);
        }
    }

    private void completeAlternative(Task<String> task, int response, int option, Request request) {
        if (task != activeGeneration) return;
        call("finishAlternative", response, option, task.getValue());
        nextAlternative(response, option, request);
    }

    private void failAlternative(Task<String> task, int response, int option, Request request) {
        if (task != activeGeneration) return;
        call("failAlternative", response, option, "Generation failed: " + message(task.getException()));
        nextAlternative(response, option, request);
    }

    private void nextAlternative(int response, int option, Request request) {
        if (option == 0) startAlternative(response, 1, request);
        else finishGeneration();
    }

    public void stop() {
        if (activeGeneration == null) return;
        Task<String> task = activeGeneration;
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
        stop();
        requests.clear();
    }

    private void finishGeneration() {
        activeGeneration = null;
        activeAlternative = -1;
        call("setGenerating", false);
        call("setStatus", ChatBackend.readyStatus());
        setBusy(false);
    }

    public void copy(String text) {
        ClipboardContent content = new ClipboardContent();
        content.putString(text);
        Clipboard.getSystemClipboard().setContent(content);
    }

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

    private static String message(Throwable error) {
        return error == null || error.getMessage() == null ? "Unknown error" : error.getMessage();
    }

    private void start(Task<?> task) {
        worker.execute(task);
    }

    private record Request(String prompt, ChatGenerationSettings settings) {}
}
