package io.github.kirstenali.deepj.chatui;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.net.URL;

public abstract class BaseChatApp extends Application {

    @Override
    public void start(Stage stage) throws Exception {
        URL fxml = getClass().getResource("/chat-view.fxml");
        URL css = getClass().getResource("/chat.css");

        if (fxml == null) {
            throw new IllegalStateException("Missing FXML file.");
        }
        if (css == null) {
            throw new IllegalStateException("Missing CSS file.");
        }

        FXMLLoader loader = new FXMLLoader(fxml);
        Scene scene = new Scene(loader.load(), 1000, 720);
        scene.getStylesheets().add(css.toExternalForm());

        ChatController controller = loader.getController();
        controller.setChatService(createChatService());

        stage.setTitle(getAppTitle());
        stage.setScene(scene);
        stage.setMinWidth(860);
        stage.setMinHeight(620);
        stage.show();
    }

    protected String getAppTitle() {
        return "DeepJ Chat";
    }

    protected abstract ChatService createChatService();
}
