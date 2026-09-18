package io.github.kirstenali.deepj.chatui;

import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;

import java.io.File;

public class ChatController {

    private ChatService chatService;

    @FXML private VBox messagesBox;
    @FXML private ScrollPane scrollPane;
    @FXML private TextArea inputArea;
    @FXML private Button sendButton;
    @FXML private HBox typingIndicator;

    @FXML private TextField modelPathField;
    @FXML private TextField temperatureField;
    @FXML private TextField topKField;
    @FXML private TextField maxTokensField;
    @FXML private TextField seedField;
    @FXML private Label statusLabel;

    public ChatController() {
    }

    public void setChatService(ChatService chatService) {
        this.chatService = chatService;
    }

    @FXML
    public void initialize() {
        setDefaultSettings();
        hideTypingIndicator();
        configureAutoScroll();

        showBotMessage("Load a model file to begin.");
        setStatus("No model loaded");
    }

    @FXML
    private void onBrowseModel() {
        if (!hasChatService()) return;
        File file = chooseModelFile();
        if (file == null) return;
        loadModel(file);
    }

    private File chooseModelFile() {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Select Model File");
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Model Files", "*.bin"));
        return chooser.showOpenDialog(getWindowOwner());
    }

    private void loadModel(File file) {
        try {
            chatService.loadModel(file.toPath());
            modelPathField.setText(file.getAbsolutePath());
            setStatus("Loaded: " + chatService.getLoadedModelName());
            showBotMessage("Model loaded: " + chatService.getLoadedModelName());
        } catch (Exception e) {
            setStatus("Failed to load model");
            showBotMessage("Error loading model: " + safeMessage(e));
        }
    }

    @FXML
    private void onSend() {
        if (!hasChatService()) return;
        String prompt = inputArea.getText().trim();
        if (prompt.isEmpty() || !hasLoadedModel()) return;
        GenerationSettings settings = readSettings();
        if (settings == null) return;
        prepareGeneration(prompt);
        Task<String> task = generationTask(prompt, settings);
        configureTaskHandlers(task);
        startBackgroundTask(task, "gpt-generate-thread");
    }

    private GenerationSettings readSettings() {
        try {
            return new GenerationSettings(
                    parseInt(maxTokensField.getText(), "Max tokens"),
                    parseFloat(temperatureField.getText(), "Temperature"),
                    parseInt(topKField.getText(), "Top-k"),
                    parseLong(seedField.getText(), "Seed"));
        } catch (IllegalArgumentException e) {
            setStatus("Invalid settings");
            showBotMessage(e.getMessage());
            return null;
        }
    }

    private void prepareGeneration(String prompt) {
        showUserMessage(prompt);
        inputArea.clear();
        setBusy(true);
        showTypingIndicator();
        setStatus("Generating...");
    }

    private Task<String> generationTask(String prompt, GenerationSettings settings) {
        return new Task<>() {
            @Override
            protected String call() {
                return chatService.generate(prompt, settings.maxTokens(), settings.temperature(),
                        settings.topK(), settings.seed());
            }
        };
    }

    private void configureTaskHandlers(Task<String> task) {
        task.setOnSucceeded(event -> {
            hideTypingIndicator();
            setBusy(false);
            showBotMessage(task.getValue());
            setStatus("Ready");
        });

        task.setOnFailed(event -> {
            hideTypingIndicator();
            setBusy(false);

            Throwable error = task.getException();
            showBotMessage("Generation failed: " + safeMessage(error));
            setStatus("Generation failed");
        });
    }

    private boolean hasChatService() {
        if (chatService != null) return true;
        showBotMessage("Chat service is not configured.");
        setStatus("Chat service not configured");
        return false;
    }

    private boolean hasLoadedModel() {
        if (chatService.isModelLoaded()) return true;
        setStatus("Load a model first");
        showBotMessage("Please load a model first.");
        return false;
    }

    @FXML
    private void onInputKeyPressed(KeyEvent event) {
        if (event.getCode() == KeyCode.ENTER && !event.isShiftDown()) {
            event.consume();
            onSend();
        }
    }

    private void startBackgroundTask(Task<?> task, String threadName) {
        Thread thread = new Thread(task, threadName);
        thread.setDaemon(true);
        thread.start();
    }

    private void setDefaultSettings() {
        temperatureField.setText("0.1");
        topKField.setText("20");
        maxTokensField.setText("200");
        seedField.setText("1234");
    }

    private void configureAutoScroll() {
        messagesBox.heightProperty().addListener((obs, oldVal, newVal) ->
                Platform.runLater(this::scrollToBottom)
        );
    }

    private void setBusy(boolean busy) {
        sendButton.setDisable(busy);
        inputArea.setDisable(busy);
    }

    private void setStatus(String text) {
        statusLabel.setText(text);
    }

    private void showTypingIndicator() {
        typingIndicator.setManaged(true);
        typingIndicator.setVisible(true);
        scrollToBottom();
    }

    private void hideTypingIndicator() {
        typingIndicator.setManaged(false);
        typingIndicator.setVisible(false);
    }

    private void showUserMessage(String text) {
        addMessage(new ChatMessage(text, true));
    }

    private void showBotMessage(String text) {
        addMessage(new ChatMessage(text, false));
    }

    private void addMessage(ChatMessage message) {
        HBox row = new HBox();
        row.setAlignment(message.user() ? Pos.CENTER_RIGHT : Pos.CENTER_LEFT);
        row.setPadding(new Insets(4, 0, 4, 0));
        row.getStyleClass().add("message-row");

        Label bubble = new Label(message.text());
        bubble.setWrapText(true);
        bubble.setMaxWidth(560);
        bubble.getStyleClass().add(message.user() ? "user-bubble" : "bot-bubble");

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        if (message.user()) {
            row.getChildren().addAll(spacer, bubble);
        } else {
            row.getChildren().addAll(bubble, spacer);
        }

        messagesBox.getChildren().add(row);
    }

    private void scrollToBottom() {
        scrollPane.layout();
        scrollPane.setVvalue(1.0);
    }

    private int parseInt(String value, String name) {
        try {
            return Integer.parseInt(value.trim());
        } catch (Exception e) {
            throw new IllegalArgumentException(name + " must be a valid integer.");
        }
    }

    private long parseLong(String value, String name) {
        try {
            return Long.parseLong(value.trim());
        } catch (Exception e) {
            throw new IllegalArgumentException(name + " must be a valid integer.");
        }
    }

    private float parseFloat(String value, String name) {
        try {
            return Float.parseFloat(value.trim());
        } catch (Exception e) {
            throw new IllegalArgumentException(name + " must be a valid number.");
        }
    }

    private String safeMessage(Throwable error) {
        if (error == null || error.getMessage() == null || error.getMessage().isBlank()) {
            return "Unknown error";
        }
        return error.getMessage();
    }

    private javafx.stage.Window getWindowOwner() {
        return inputArea.getScene().getWindow();
    }

    private record GenerationSettings(int maxTokens, float temperature, int topK, long seed) {}
}
