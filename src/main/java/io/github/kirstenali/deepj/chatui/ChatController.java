package io.github.kirstenali.deepj.chatui;

import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;

import java.io.File;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

public class ChatController {

    private ChatService chatService;
    private ChatAppearance appearance;
    private boolean busy;

    @FXML private BorderPane rootPane;
    @FXML private VBox messagesBox;
    @FXML private VBox welcomeBox;
    @FXML private VBox settingsPanel;
    @FXML private Region settingsBackdrop;
    @FXML private ScrollPane scrollPane;
    @FXML private TextArea inputArea;
    @FXML private Button sendButton;
    @FXML private Button newChatButton;
    @FXML private Button modelButton;
    @FXML private Button lightThemeButton;
    @FXML private Button darkThemeButton;
    @FXML private Button midnightThemeButton;
    @FXML private HBox typingIndicator;
    @FXML private TextField temperatureField;
    @FXML private TextField topKField;
    @FXML private TextField maxTokensField;
    @FXML private TextField seedField;
    @FXML private Label modelNameLabel;
    @FXML private Label modelDetailsLabel;
    @FXML private Label modelStatusDot;
    @FXML private Label statusLabel;

    public void setChatService(ChatService service) {
        chatService = service;
        updateSendState();
    }

    @FXML
    public void initialize() {
        setDefaultSettings();
        configureAppearance();
        configureAutoScroll();
        hideTypingIndicator();
        setStatus("Choose a DeepJ Prism checkpoint to begin");
        updateSendState();
    }

    @FXML
    private void onBrowseModel() {
        appearance.hide();
        File file = chooseModelFile();
        if (file != null && chatService != null) loadModel(file.toPath());
    }

    @FXML
    private void onNewChat() {
        messagesBox.getChildren().clear();
        showWelcome(true);
        appearance.hide();
        inputArea.clear();
        inputArea.requestFocus();
    }

    @FXML
    private void onToggleSettings() {
        appearance.toggle();
    }

    @FXML
    private void onToggleSidebar() {
        appearance.toggleSidebar();
    }

    @FXML
    private void onTheme(ActionEvent event) {
        Button button = (Button) event.getSource();
        appearance.apply(ChatTheme.valueOf(button.getUserData().toString()));
    }

    @FXML
    private void onSend() {
        String prompt = inputArea.getText().trim();
        GenerationSettings settings = settings(prompt);
        if (settings != null) startGeneration(prompt, settings);
    }

    @FXML
    private void onInputKeyPressed(KeyEvent event) {
        if (event.getCode() != KeyCode.ENTER || event.isShiftDown()) return;
        event.consume();
        onSend();
    }

    private File chooseModelFile() {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Choose a DeepJ model");
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("DeepJ model", "*.dj"));
        return chooser.showOpenDialog(inputArea.getScene().getWindow());
    }

    private void loadModel(Path path) {
        setBusy(true);
        showModelState("Loading model…", path.getFileName().toString(), "model-loading");
        setStatus("Loading model…");
        Task<Void> task = modelTask(path);
        configureModelTask(task);
        ChatRuntime.start(task, "deepj-model-loader");
    }

    private Task<Void> modelTask(Path path) {
        return new Task<>() {
            @Override
            protected Void call() throws Exception {
                chatService.loadModel(path);
                return null;
            }
        };
    }

    private void configureModelTask(Task<Void> task) {
        task.setOnSucceeded(event -> modelLoaded());
        task.setOnFailed(event -> modelLoadFailed(task.getException()));
    }

    private void modelLoaded() {
        String name = chatService.getLoadedModelName();
        showModelState(name, "Ready on " + ChatRuntime.backendName(), "model-ready");
        setStatus("Ready on " + ChatRuntime.backendName());
        setBusy(false);
        inputArea.requestFocus();
    }

    private void modelLoadFailed(Throwable error) {
        setBusy(false);
        showModelState("Model unavailable", "Choose another checkpoint", "model-error");
        setStatus("Could not load model");
        showWelcome(false);
        addMessage(new ChatMessage("Model loading failed: "
                + ChatRuntime.errorMessage(error), false));
    }

    private GenerationSettings settings(String prompt) {
        if (!canGenerate(prompt)) return null;
        try {
            return parsedSettings();
        } catch (IllegalArgumentException error) {
            setStatus(error.getMessage());
            return null;
        }
    }

    private boolean canGenerate(String prompt) {
        if (busy || prompt.isEmpty()) return false;
        if (chatService != null && chatService.isModelLoaded()) return true;
        setStatus("Choose a model first");
        return false;
    }

    private GenerationSettings parsedSettings() {
        return GenerationSettings.parse(maxTokensField.getText(), temperatureField.getText(),
                topKField.getText(), seedField.getText());
    }

    private void startGeneration(String prompt, GenerationSettings settings) {
        appearance.hide();
        showWelcome(false);
        addMessage(new ChatMessage(prompt, true));
        Label response = addMessage(new ChatMessage("", false));
        prepareComposer();
        Task<String> task = generationTask(prompt, settings, response);
        configureGenerationTask(task, response);
        ChatRuntime.start(task, "deepj-generation");
    }

    private void prepareComposer() {
        inputArea.clear();
        setBusy(true);
        showTypingIndicator();
        setStatus("Generating…");
    }

    private Task<String> generationTask(String prompt, GenerationSettings settings, Label response) {
        Consumer<String> listener = streamTo(response);
        return new Task<>() {
            @Override
            protected String call() {
                return chatService.generate(prompt, settings.maxTokens(), settings.temperature(),
                        settings.topK(), settings.seed(), listener);
            }
        };
    }

    private Consumer<String> streamTo(Label response) {
        AtomicBoolean started = new AtomicBoolean();
        return text -> Platform.runLater(() -> updateStream(response, text, started));
    }

    private void updateStream(Label response, String text, AtomicBoolean started) {
        if (started.compareAndSet(false, true)) hideTypingIndicator();
        response.setText(text);
        scrollToBottom();
    }

    private void configureGenerationTask(Task<String> task, Label response) {
        task.setOnSucceeded(event -> generationFinished(response, task.getValue()));
        task.setOnFailed(event -> generationFailed(response, task.getException()));
    }

    private void generationFinished(Label response, String text) {
        hideTypingIndicator();
        response.setText(text.isBlank() ? "The model returned no text." : text);
        setBusy(false);
        setStatus("Ready on " + ChatRuntime.backendName());
        inputArea.requestFocus();
    }

    private void generationFailed(Label response, Throwable error) {
        hideTypingIndicator();
        response.setText("Generation failed: " + ChatRuntime.errorMessage(error));
        setBusy(false);
        setStatus("Generation failed");
    }

    private Label addMessage(ChatMessage message) {
        ChatMessageView view = ChatMessageView.create(message);
        messagesBox.getChildren().add(view.row());
        scrollToBottom();
        return view.content();
    }

    private void configureAppearance() {
        appearance = new ChatAppearance(rootPane, settingsPanel, settingsBackdrop,
                lightThemeButton, darkThemeButton, midnightThemeButton);
        appearance.apply(ChatTheme.LIGHT);
    }

    private void configureAutoScroll() {
        messagesBox.heightProperty().addListener((value, oldHeight, newHeight) ->
                Platform.runLater(this::scrollToBottom));
    }

    private void setDefaultSettings() {
        temperatureField.setText("0.7");
        topKField.setText("20");
        maxTokensField.setText("120");
        seedField.setText("2026");
    }

    private void setBusy(boolean value) {
        busy = value;
        inputArea.setDisable(value);
        newChatButton.setDisable(value);
        modelButton.setDisable(value);
        updateSendState();
    }

    private void updateSendState() {
        boolean unloaded = chatService == null || !chatService.isModelLoaded();
        sendButton.setDisable(busy || unloaded);
    }

    private void showWelcome(boolean visible) {
        welcomeBox.setManaged(visible);
        welcomeBox.setVisible(visible);
    }

    private void showModelState(String name, String details, String style) {
        ModelCardState.show(modelNameLabel, modelDetailsLabel, modelStatusDot,
                name, details, style);
    }

    private void showTypingIndicator() {
        typingIndicator.setManaged(true);
        typingIndicator.setVisible(true);
    }

    private void hideTypingIndicator() {
        typingIndicator.setManaged(false);
        typingIndicator.setVisible(false);
    }

    private void scrollToBottom() {
        scrollPane.setVvalue(1.0);
    }

    private void setStatus(String text) {
        statusLabel.setText(text);
    }

}
