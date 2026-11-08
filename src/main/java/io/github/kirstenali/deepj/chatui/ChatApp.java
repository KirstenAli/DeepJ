package io.github.kirstenali.deepj.chatui;

import javafx.stage.Stage;

public class ChatApp extends BaseChatApp {
    private Stage stage;

    @Override
    public void start(Stage window) {
        stage = window;
        super.start(window);
    }

    @Override
    protected ChatService createChatService() {
        return new AgentChatService(new DesktopCommandApproval(stage)::review);
    }

    public static void main(String[] args) {
        launch(args);
    }
}
