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
        if (chatService == null) {
            showBotMessage("Chat service is not configured.");
            setStatus("Chat service not configured");
            return;
        }

        FileChooser chooser = new FileChooser();
        chooser.setTitle("Select Model File");
        chooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("Model Files", "*.bin")
        );

        File file = chooser.showOpenDialog(getWindowOwner());
        if (file == null) {
            return;
        }

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
        if (chatService == null) {
            showBotMessage("Chat service is not configured.");
            setStatus("Chat service not configured");
            return;
        }

        String prompt = inputArea.getText().trim();
        if (prompt.isEmpty()) {
            return;
        }

        if (!chatService.isModelLoaded()) {
            setStatus("Load a model first");
            showBotMessage("Please load a model first.");
            return;
        }

        int maxTokens;
        float temperature;
        int topK;
        long seed;

        try {
            maxTokens = parseInt(maxTokensField.getText(), "Max tokens");
            temperature = parseFloat(temperatureField.getText(), "Temperature");
            topK = parseInt(topKField.getText(), "Top-k");
            seed = parseLong(seedField.getText(), "Seed");
        } catch (IllegalArgumentException e) {
            setStatus("Invalid settings");
            showBotMessage(e.getMessage());
            return;
        }

        showUserMessage(prompt);
        inputArea.clear();

        setBusy(true);
        showTypingIndicator();
        setStatus("Generating...");

        Task<String> task = new Task<>() {
            @Override
            protected String call() {
                return chatService.generate(prompt, maxTokens, temperature, topK, seed);
            }
        };

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

        startBackgroundTask(task, "gpt-generate-thread");
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
}
