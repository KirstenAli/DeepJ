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
import java.util.Objects;

public final class ChatWindow {

    private final Stage stage;
    private final WebView view;
    private final ChatService service;
    private boolean busy;
    private int nextResponse;

    ChatWindow(Stage stage, WebView view, ChatService service) {
        this.stage = stage;
        this.view = view;
        this.service = service;
    }

    void open() {
        view.setContextMenuEnabled(false);
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
        call("setModel", service.getLoadedModelName());
        call("setStatus", "Ready on CPU");
        setBusy(false);
    }

    private void loadFailed(Throwable error) {
        call("setStatus", "Could not load model: " + message(error));
        setBusy(false);
    }

    public void send(String prompt, int maxTokens, double temperature, int topK, long seed) {
        if (busy || !service.isModelLoaded()) return;
        if (prompt == null || prompt.isBlank()) return;
        try {
            startGeneration(prompt.trim(), new Settings(maxTokens, temperature, topK, seed));
        } catch (IllegalArgumentException error) {
            call("setStatus", error.getMessage());
        }
    }

    private void startGeneration(String prompt, Settings settings) {
        int response = nextResponse++;
        call("startResponse", prompt, response);
        setBusy(true);
        Task<String> task = generation(prompt, settings);
        task.valueProperty().addListener((ob, old, text) -> partial(response, text));
        task.setOnSucceeded(event -> complete(response, task.getValue()));
        task.setOnFailed(event -> failed(response, task.getException()));
        start(task);
    }

    private Task<String> generation(String prompt, Settings settings) {
        return new Task<>() {
            @Override protected String call() {
                return service.generate(prompt, settings.maxTokens(), (float) settings.temperature(),
                        settings.topK(), settings.seed(), this::updateValue);
            }
        };
    }

    private void partial(int response, String text) {
        if (text != null && !text.isEmpty()) call("updateResponse", response, text);
    }

    private void complete(int response, String text) {
        call("finishResponse", response, text == null ? "" : text);
        call("setStatus", "Ready on CPU");
        setBusy(false);
    }

    private void failed(int response, Throwable error) {
        call("failResponse", response, "Generation failed: " + message(error));
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

    private static void start(Task<?> task) {
        Thread thread = new Thread(task, "deepj-chat-worker");
        thread.setDaemon(true);
        thread.start();
    }

    private record Settings(int maxTokens, double temperature, int topK, long seed) {
        private Settings {
            if (maxTokens < 1) throw new IllegalArgumentException("Maximum tokens must be positive");
            if (!Double.isFinite(temperature) || temperature <= 0) {
                throw new IllegalArgumentException("Temperature must be positive");
            }
            if (topK < 0) throw new IllegalArgumentException("Top-k cannot be negative");
        }
    }
}
