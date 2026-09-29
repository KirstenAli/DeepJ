package io.github.kirstenali.deepj.chatui;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;
import java.net.URL;

public abstract class BaseChatApp extends Application {

    @Override
    public void start(Stage stage) throws Exception {
        URL fxml = getClass().getResource("/chat-view.fxml");
        URL css = getClass().getResource("/chat.css");
        requireResource(fxml, "FXML");
        requireResource(css, "CSS");
        FXMLLoader loader = new FXMLLoader(fxml);
        Scene scene = loadScene(loader, css);
        configureController(loader);
        showStage(stage, scene);
    }

    private static void requireResource(URL resource, String name) {
        if (resource == null) throw new IllegalStateException("Missing " + name + " file.");
    }

    private Scene loadScene(FXMLLoader loader, URL css) throws IOException {
        Scene scene = new Scene(loader.load(), 1000, 720);
        scene.getStylesheets().add(css.toExternalForm());
        return scene;
    }

    private void configureController(FXMLLoader loader) {
        ChatController controller = loader.getController();
        controller.setChatService(createChatService());
    }

    private void showStage(Stage stage, Scene scene) {
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
