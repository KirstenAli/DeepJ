package io.github.kirstenali.deepj.chatui;

import javafx.application.Platform;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ToolBar;
import javafx.scene.layout.BorderPane;
import javafx.stage.DirectoryChooser;
import javafx.stage.Stage;

/** Separate desktop launcher with optional agent mode in the existing DeepJ chat window. */
public final class AgentDesktopApp extends BaseChatApp {
    private final CheckBox agentMode = new CheckBox("Agent mode");
    private final Button directory = new Button("Working directory…");
    private Stage stage;
    private AgentChatService service;

    public static void main(String[] args) { launch(args); }

    @Override
    public void start(Stage window) {
        stage = window;
        agentMode.setDisable(true);
        directory.setDisable(true);
        super.start(window);
        agentMode.selectedProperty().addListener((ob, old, selected) -> service.enable(selected));
        directory.setOnAction(event -> chooseDirectory());
        var scene = window.getScene();
        var chat = scene.getRoot();
        var layout = new BorderPane();
        scene.setRoot(layout);
        layout.setCenter(chat);
        layout.setTop(new ToolBar(agentMode, directory));
    }

    @Override
    protected ChatService createChatService() {
        service = new AgentChatService(this::modelChanged, new DesktopCommandApproval(stage)::approve);
        return service;
    }

    @Override
    protected String getAppTitle() { return "DeepJ Agent"; }

    private void modelChanged(boolean supported) {
        Platform.runLater(() -> {
            agentMode.setSelected(false);
            agentMode.setDisable(!supported);
            directory.setDisable(!supported);
        });
    }

    private void chooseDirectory() {
        var chooser = new DirectoryChooser();
        chooser.setTitle("Choose the terminal working directory");
        if (service.directory().toFile().isDirectory()) chooser.setInitialDirectory(service.directory().toFile());
        var selected = chooser.showDialog(stage);
        if (selected != null) service.directory(selected.toPath());
    }
}
