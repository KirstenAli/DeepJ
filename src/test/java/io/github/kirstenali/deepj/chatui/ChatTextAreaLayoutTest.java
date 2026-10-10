package io.github.kirstenali.deepj.chatui;

import javafx.application.Platform;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;
import javafx.stage.Stage;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

@EnabledIfSystemProperty(named = "deepj.uiTests", matches = "true")
class ChatTextAreaLayoutTest {

    private static final String SAMPLE = "was so happy that he got her mom and said, "
            + "\"Lily, I want to do you want to play with my toy and it and be careful for them. "
            + "They played with the mouse. They played with the curt and the other and the little boy "
            + "had a good idea. He was not scared. She was so happy.";
    private ChatTextArea area;
    private HBox row;
    private Stage stage;

    @BeforeAll
    static void startJavaFx() {
        Platform.startup(() -> Platform.setImplicitExit(false));
    }

    @AfterAll
    static void stopJavaFx() {
        Platform.exit();
    }

    @Test
    void wrappedResponseFitsItsViewport() throws Exception {
        runOnFxThread(() -> show(false));
        assertTextFits();
        runOnFxThread(this::checkCopyButtonPosition);
        runOnFxThread(() -> area.setText(SAMPLE + SAMPLE));
        assertTextFits();
        runOnFxThread(this::checkCopyButtonPosition);
        runOnFxThread(stage::close);
    }

    @Test
    void wrappedUserMessageFitsItsViewport() throws Exception {
        runOnFxThread(() -> show(true));
        assertTextFits();
        runOnFxThread(stage::close);
    }

    @Test
    void responseFitsInsideConversationLayout() throws Exception {
        runOnFxThread(this::showConversation);
        assertTextFits();
        runOnFxThread(this::checkCopyButtonPosition);
        runOnFxThread(stage::close);
    }

    private void show(boolean user) {
        ChatMessageView view = ChatMessageView.create(new ChatMessage(SAMPLE, user));
        area = (ChatTextArea) view.content();
        row = view.row();
        stage = new Stage();
        stage.setScene(scene(row));
        stage.show();
    }

    private void showConversation() {
        try {
            Parent root = FXMLLoader.load(getClass().getResource("/chat-view.fxml"));
            ChatMessageView view = ChatMessageView.create(new ChatMessage(SAMPLE, false));
            area = (ChatTextArea) view.content();
            row = view.row();
            stage = new Stage();
            stage.setScene(styledScene(root));
            root.applyCss();
            ((VBox) root.lookup(".messages-box")).getChildren().add(row);
            stage.show();
        } catch (Exception error) {
            throw new IllegalStateException(error);
        }
    }

    private Scene styledScene(Parent root) {
        Scene scene = new Scene(root, 1200, 800);
        scene.getStylesheets().add(getClass().getResource("/chat.css").toExternalForm());
        return scene;
    }

    private void layout() {
        stage.getScene().getRoot().applyCss();
        stage.getScene().getRoot().layout();
    }

    private Scene scene(HBox row) {
        VBox root = new VBox(row);
        root.getStyleClass().addAll("app-root", "theme-light");
        Scene scene = new Scene(root, 900, 400);
        scene.getStylesheets().add(getClass().getResource("/chat.css").toExternalForm());
        return scene;
    }

    private void assertTextFits() throws Exception {
        runOnFxThread(this::layout);
        runOnFxThread(this::checkViewport);
    }

    private void checkViewport() {
        Text text = (Text) area.lookup(".text");
        Region viewport = (Region) area.lookup(".viewport");
        Assertions.assertTrue(viewport.getHeight() >= text.getLayoutBounds().getHeight(),
                () -> "Text height " + text.getLayoutBounds().getHeight()
                        + " exceeds viewport " + viewport.getHeight());
    }

    private void checkCopyButtonPosition() {
        Button copy = (Button) row.lookup(".copy-message-button");
        Text text = (Text) area.lookup(".text");
        double textBottom = text.localToScene(text.getBoundsInLocal()).getMaxY();
        double copyTop = copy.localToScene(copy.getBoundsInLocal()).getMinY();
        Assertions.assertTrue(copyTop >= textBottom && copyTop - textBottom < 12,
                () -> "gap=" + (copyTop - textBottom) + " area=" + area.getHeight()
                        + " pref=" + area.getPrefHeight() + " text=" + text.getLayoutBounds().getHeight());
    }

    private void runOnFxThread(Runnable action) throws Exception {
        CountDownLatch done = new CountDownLatch(1);
        AtomicReference<Throwable> failure = new AtomicReference<>();
        Platform.runLater(() -> runAction(action, failure, done));
        if (!done.await(10, TimeUnit.SECONDS)) throw new AssertionError("JavaFX timed out");
        if (failure.get() != null) throw new AssertionError(failure.get());
    }

    private void runAction(Runnable action, AtomicReference<Throwable> failure,
                           CountDownLatch done) {
        try {
            action.run();
        } catch (Throwable error) {
            failure.set(error);
        }
        done.countDown();
    }
}
