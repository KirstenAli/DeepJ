package io.github.kirstenali.deepj.chatui;

import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;

import java.io.File;

public final class ChatController {

    private ChatService chatService;
    private boolean busy;

    @FXML private BorderPane rootPane;
    @FXML private VBox sidebar;
    @FXML private VBox settingsPane;
    @FXML private ChatTranscript transcript;
    @FXML private TextArea inputArea;
    @FXML private Button sendButton;
    @FXML private Button modelButton;
    @FXML private ComboBox<String> themeSelector;
    @FXML private TextField temperatureField;
    @FXML private TextField topKField;
    @FXML private TextField maxTokensField;
    @FXML private TextField seedField;
    @FXML private Label modelNameLabel;
    @FXML private Label statusLabel;

    public void setChatService(ChatService service) {
        chatService = service;
    }

    @FXML
    private void initialize() {
        temperatureField.setText("0.8");
        topKField.setText("40");
        maxTokensField.setText("80");
        seedField.setText("1234");
        themeSelector.getItems().addAll("Light", "Dark", "Warm", "Ocean");
        themeSelector.getSelectionModel().select("Light");
        themeSelector.valueProperty().addListener((ob, old, name) -> applyTheme(name));
    }

    @FXML
    private void onBrowseModel() {
        if (busy || chatService == null) return;
        File file = chooseModel();
        if (file != null) loadModel(file);
    }

    private File chooseModel() {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Choose a DeepJ model");
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("DeepJ models", "*.dj", "*.bin"));
        return chooser.showOpenDialog(rootPane.getScene().getWindow());
    }

    private void loadModel(File file) {
        setBusy(true);
        setStatus("Loading " + file.getName() + "…");
        Task<Void> task = new Task<>() {
            @Override protected Void call() throws Exception {
                chatService.loadModel(file.toPath());
                return null;
            }
        };
        task.setOnSucceeded(event -> modelLoaded());
        task.setOnFailed(event -> showFailure("Could not load model", task.getException()));
        start(task);
    }

    private void modelLoaded() {
        modelNameLabel.setText(chatService.getLoadedModelName());
        transcript.clear();
        setBusy(false);
        setStatus("Ready on CPU");
    }

    @FXML
    private void onSend() {
        if (busy || !hasModel()) return;
        String prompt = inputArea.getText().trim();
        if (prompt.isEmpty()) return;
        Settings settings = readSettings();
        if (settings == null) return;
        startGeneration(prompt, settings);
    }

    private void startGeneration(String prompt, Settings settings) {
        transcript.add(true, prompt);
        int response = transcript.add(false, "Generating…");
        inputArea.clear();
        setBusy(true);
        setStatus("Generating…");
        Task<String> task = generation(prompt, settings);
        task.valueProperty().addListener((ob, old, text) -> showPartial(response, text));
        task.setOnSucceeded(event -> showComplete(response, task.getValue()));
        task.setOnFailed(event -> showFailure(response, task.getException()));
        start(task);
    }

    private Task<String> generation(String prompt, Settings settings) {
        return new Task<>() {
            @Override protected String call() {
                return chatService.generate(prompt, settings.maxTokens(), settings.temperature(),
                        settings.topK(), settings.seed(), this::updateValue);
            }
        };
    }

    private void showPartial(int response, String text) {
        if (text != null && !text.isEmpty()) transcript.update(response, text);
    }

    private void showComplete(int response, String text) {
        transcript.update(response, text == null || text.isBlank()
                ? "No new text was generated. Try a different prompt." : text);
        setBusy(false);
        setStatus("Ready on CPU");
    }

    private void showFailure(int response, Throwable error) {
        transcript.update(response, "Generation failed: " + message(error));
        setBusy(false);
        setStatus("Generation failed");
    }

    private void showFailure(String label, Throwable error) {
        setBusy(false);
        setStatus(label + ": " + message(error));
    }

    private static String message(Throwable error) {
        return error == null || error.getMessage() == null ? "Unknown error" : error.getMessage();
    }

    private boolean hasModel() {
        if (chatService != null && chatService.isModelLoaded()) return true;
        setStatus("Choose a model first");
        return false;
    }

    private Settings readSettings() {
        try {
            return new Settings(Integer.parseInt(maxTokensField.getText().trim()),
                    Float.parseFloat(temperatureField.getText().trim()),
                    Integer.parseInt(topKField.getText().trim()),
                    Long.parseLong(seedField.getText().trim()));
        } catch (IllegalArgumentException error) {
            setStatus("Check generation settings for valid values");
            return null;
        }
    }

    @FXML
    private void onInputKeyPressed(KeyEvent event) {
        if (event.getCode() != KeyCode.ENTER || event.isShiftDown()) return;
        event.consume();
        onSend();
    }

    @FXML
    private void onNewChat() {
        if (busy) return;
        transcript.clear();
        inputArea.clear();
        setStatus(hasModel() ? "Ready on CPU" : "Choose a model to begin");
    }

    @FXML
    private void onToggleSidebar() {
        boolean show = !sidebar.isVisible();
        sidebar.setVisible(show);
        sidebar.setManaged(show);
    }

    @FXML
    private void onToggleSettings() {
        boolean show = !settingsPane.isVisible();
        settingsPane.setVisible(show);
        settingsPane.setManaged(show);
    }

    private void applyTheme(String name) {
        if (name == null) return;
        rootPane.getStyleClass().removeIf(style -> style.startsWith("theme-"));
        rootPane.getStyleClass().add("theme-" + name.toLowerCase());
        transcript.setTheme(name.toLowerCase());
    }

    private void setBusy(boolean value) {
        busy = value;
        inputArea.setDisable(value);
        sendButton.setDisable(value);
        modelButton.setDisable(value);
    }

    private void setStatus(String text) {
        statusLabel.setText(text);
    }

    private static void start(Task<?> task) {
        Thread thread = new Thread(task, "deepj-chat-worker");
        thread.setDaemon(true);
        thread.start();
    }

    private record Settings(int maxTokens, float temperature, int topK, long seed) {
        private Settings {
            if (maxTokens < 1) throw new IllegalArgumentException("maxTokens");
            if (!Float.isFinite(temperature) || temperature <= 0) throw new IllegalArgumentException("temperature");
            if (topK < 0) throw new IllegalArgumentException("topK");
        }
    }
}
