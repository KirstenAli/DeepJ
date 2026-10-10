package io.github.kirstenali.deepj.chatui;

import javafx.application.Platform;
import javafx.concurrent.Worker;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.web.WebView;
import javafx.stage.Stage;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@EnabledIfSystemProperty(named = "deepj.uiTests", matches = "true")
class ChatTranscriptTest {

    @BeforeAll
    static void startJavaFx() {
        Platform.startup(() -> {});
        Platform.setImplicitExit(false);
    }

    @Test
    void messagesWrapAndRemainSelectable() throws Exception {
        CountDownLatch loaded = new CountDownLatch(1);
        AtomicReference<Stage> stage = new AtomicReference<>();
        AtomicReference<ChatTranscript> transcript = new AtomicReference<>();
        Platform.runLater(() -> show(stage, transcript, loaded));
        assertTrue(loaded.await(10, TimeUnit.SECONDS));
        assertMessages(stage.get(), transcript.get());
    }

    @Test
    void chatViewLoadsWithTranscriptAndThemes() throws Exception {
        CountDownLatch checked = new CountDownLatch(1);
        AtomicReference<Throwable> failure = new AtomicReference<>();
        Platform.runLater(() -> loadView(checked, failure));
        assertTrue(checked.await(10, TimeUnit.SECONDS));
        if (failure.get() != null) throw new AssertionError(failure.get());
    }

    private static void loadView(CountDownLatch checked, AtomicReference<Throwable> failure) {
        try {
            Parent root = FXMLLoader.load(ChatTranscriptTest.class.getResource("/chat-view.fxml"));
            assertTrue(root.lookupAll(".theme-picker").size() == 1);
            assertTrue(root.lookupAll(".chat-input").size() == 1);
        } catch (Throwable error) {
            failure.set(error);
        } finally {
            checked.countDown();
        }
    }

    private static void show(AtomicReference<Stage> stage,
                             AtomicReference<ChatTranscript> transcript, CountDownLatch loaded) {
        ChatTranscript chat = new ChatTranscript();
        Stage window = new Stage();
        window.setScene(new Scene(chat, 540, 420));
        window.show();
        stage.set(window);
        transcript.set(chat);
        WebView view = (WebView) chat.getChildren().get(0);
        view.getEngine().getLoadWorker().stateProperty().addListener((ob, old, state) -> {
            if (state == Worker.State.SUCCEEDED) loaded.countDown();
        });
        if (view.getEngine().getLoadWorker().getState() == Worker.State.SUCCEEDED) loaded.countDown();
    }

    private static void assertMessages(Stage stage, ChatTranscript transcript) throws Exception {
        CountDownLatch checked = new CountDownLatch(1);
        AtomicReference<Throwable> failure = new AtomicReference<>();
        Platform.runLater(() -> checkOnUi(stage, transcript, checked, failure));
        assertTrue(checked.await(10, TimeUnit.SECONDS));
        if (failure.get() != null) throw new AssertionError(failure.get());
    }

    private static void checkOnUi(Stage stage, ChatTranscript transcript,
                                  CountDownLatch checked, AtomicReference<Throwable> failure) {
        try {
            checkContent(transcript);
        } catch (Throwable error) {
            failure.set(error);
        } finally {
            stage.close();
            checked.countDown();
        }
    }

    private static void checkContent(ChatTranscript transcript) {
        String answer = "A selectable answer with enough words to wrap naturally ".repeat(8);
        transcript.add(true, "A long question about the model and its training data ".repeat(10));
        transcript.add(false, "Generating…");
        transcript.update(1, answer);
        WebView view = (WebView) transcript.getChildren().get(0);
        Object text = view.getEngine().executeScript("document.querySelector('#message-1 .content').textContent");
        Object wrapped = view.getEngine().executeScript("document.querySelector('#message-0').offsetHeight > 60");
        Object selected = view.getEngine().executeScript("""
                var range = document.createRange();
                range.selectNodeContents(document.querySelector('#message-1 .content'));
                window.getSelection().removeAllRanges();
                window.getSelection().addRange(range);
                window.getSelection().toString();
                """);
        assertEquals(answer, text);
        assertEquals(true, wrapped);
        assertEquals(answer, selected);
        assertNaturalHeight(view);
    }

    private static void assertNaturalHeight(WebView view) {
        Object clipped = view.getEngine().executeScript("""
                [...document.querySelectorAll('.content')].some(
                  node => node.scrollHeight > node.clientHeight + 1)
                """);
        assertEquals(false, clipped);
    }
}
