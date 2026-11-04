package io.github.kirstenali.deepj.chatui;

import javafx.stage.Stage;

/** Separate desktop launcher with optional agent mode in the existing DeepJ chat window. */
public final class AgentDesktopApp extends BaseChatApp {
    private Stage stage;

    public static void main(String[] args) { launch(args); }

    @Override
    public void start(Stage window) {
        stage = window;
        super.start(window);
    }

    @Override
    protected ChatService createChatService() {
        return new AgentChatService(new DesktopCommandApproval(stage)::review);
    }

    @Override
    protected String getAppTitle() { return "DeepJ Agent"; }
}
