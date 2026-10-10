package io.github.kirstenali.deepj.chatui;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.web.WebView;
import javafx.stage.Stage;

public abstract class BaseChatApp extends Application {

    private ChatWindow chatWindow;

    @Override
    public void start(Stage stage) {
        WebView view = new WebView();
        chatWindow = new ChatWindow(stage, view, createChatService());
        stage.setScene(new Scene(view, 1100, 760));
        showStage(stage);
        chatWindow.open();
    }

    private void showStage(Stage stage) {
        stage.setTitle(getAppTitle());
        stage.setMinWidth(760);
        stage.setMinHeight(560);
        stage.show();
    }

    protected String getAppTitle() {
        return "DeepJ Chat";
    }

    protected abstract ChatService createChatService();
}
