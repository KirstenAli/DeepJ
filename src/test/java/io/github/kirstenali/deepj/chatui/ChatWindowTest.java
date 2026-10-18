package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.chatui.ChatWindowTestSupport.AlternativeService;
import io.github.kirstenali.deepj.chatui.ChatWindowTestSupport.BlockingService;
import io.github.kirstenali.deepj.chatui.ChatWindowTestSupport.Invocation;
import javafx.application.Platform;
import javafx.concurrent.Worker;
import javafx.event.Event;
import javafx.scene.Scene;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.web.WebView;
import javafx.stage.Stage;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

@EnabledIfSystemProperty(named = "deepj.uiTests", matches = "true")
class ChatWindowTest {

    @BeforeAll
    static void startJavaFx() {
        Platform.startup(() -> {});
        Platform.setImplicitExit(false);
    }

    @Test
    void messagesAreSelectableAndNotClipped() throws Exception {
        Fixture fixture = open();
        verify(fixture, view -> {
            script(view, "startResponse('A long question '.repeat(45), 1)");
            script(view, "updateResponse(1, 'A selectable answer '.repeat(45))");
            String clipped = "[...document.querySelectorAll('.message-content')]"
                    + ".some(n => n.scrollHeight > n.clientHeight + 1)";
            assertEquals(false, script(view, clipped));
            assertEquals(true, script(view, "document.querySelector('.message.user').offsetHeight > 50"));
            assertEquals(1, ((Number) script(view, "document.querySelectorAll('#conversation').length")).intValue());
            assertSelection(view);
        });
    }

    @Test
    void settingsAndEmptyResponsesBehaveClearly() throws Exception {
        Fixture fixture = open();
        verify(fixture, view -> {
            assertEquals(true, script(view, "document.getElementById('settingsBackdrop').hidden"));
            assertEquals("1", script(view, "document.getElementById('repetitionPenalty').value"));
            script(view, "document.getElementById('openSettings').click()");
            assertEquals(false, script(view, "document.getElementById('settingsBackdrop').hidden"));
            assertEquals("closeSettings", script(view, "document.activeElement.id"));
            script(view, "document.querySelector('[data-theme=dark]').click()");
            assertEquals("theme-dark", script(view, "document.body.className"));
            script(view, "document.getElementById('closeSettings').click()");
            assertEquals("openSettings", script(view, "document.activeElement.id"));
            script(view, "startResponse('Hello', 2)");
            script(view, "finishResponse(2, '')");
            assertEquals(0, ((Number) script(view, "document.querySelectorAll('.assistant').length")).intValue());
            assertEquals(true, script(view, "document.getElementById('toast').classList.contains('visible')"));
        });
    }

    @Test
    void copyAppearsOnlyForGeneratedText() throws Exception {
        Fixture fixture = open();
        verify(fixture, view -> {
            script(view, "startResponse('Hello', 4)");
            assertEquals(0, copyCount(view));
            script(view, "updateResponse(4, 'Answer')");
            assertEquals(1, copyCount(view));
            script(view, "failResponse(4, 'Failure')");
            assertEquals(0, copyCount(view));
        });
    }

    @Test
    void composerGrowsAndInvalidSettingsDoNotSend() throws Exception {
        Fixture fixture = open();
        verify(fixture, view -> {
            script(view, "setModel('stub.dj')");
            script(view, "document.getElementById('prompt').value = 'one\\ntwo\\nthree'; resizePrompt()");
            assertTrue(styleNumber(view, "#prompt", "height") > 50);
            script(view, "document.getElementById('maxTokens').value = '0'; submitPrompt()");
            assertEquals(0, ((Number) script(view, "document.querySelectorAll('.user').length")).intValue());
            assertEquals(false, script(view, "document.getElementById('settingsBackdrop').hidden"));
            assertEquals(true, script(view, "document.getElementById('generationOptions').open"));
            script(view, "document.getElementById('maxTokens').value = '80';"
                    + "document.getElementById('repetitionPenalty').value = '0.9'; submitPrompt()");
            assertEquals(0, ((Number) script(view, "document.querySelectorAll('.user').length")).intValue());
            script(view, "newChat()");
            assertTrue(styleNumber(view, "#prompt", "height") <= 31);
        });
    }

    private static int copyCount(WebView view) {
        return ((Number) script(view, "document.querySelectorAll('.copy-button').length")).intValue();
    }

    @Test
    void welcomeAndMessageTextHaveReadableSpacing() throws Exception {
        Fixture fixture = open();
        verify(fixture, view -> {
            assertTrue(styleNumber(view, "#welcome p", "marginBottom") >= 24);
            assertTrue(styleNumber(view, ".disclaimer", "marginTop") >= 15);
            assertNull(script(view, "document.getElementById('sidebarModel')"));
            assertNull(script(view, "document.getElementById('modelBadge')"));
            assertEquals(true, script(view, "!!document.getElementById('composerModel')"));
            script(view, "startResponse('Hello', 3)");
            assertTrue(styleNumber(view, ".message", "fontSize") >= 16);
        });
    }

    @Test
    void desktopBridgeStreamsIntoConversation() throws Exception {
        Fixture fixture = open();
        verifyWithoutClosing(fixture, view -> {
            script(view, "setModel('stub.dj')");
            script(view, "document.getElementById('repetitionPenalty').value = '1.2'");
            script(view, "document.getElementById('prompt').value = 'Hello'");
            script(view, "document.getElementById('prompt').focus()");
            pressEnter(view, true);
            assertEquals(0, ((Number) script(view, "document.querySelectorAll('.user').length")).intValue());
            pressEnter(view, false);
            assertEquals("Hello", script(view, "document.querySelector('.user .message-content').textContent"));
            assertEquals("", script(view, "document.getElementById('prompt').value"));
        });
        assertTrue(((TestService) fixture.service).generated.await(10, TimeUnit.SECONDS));
        assertEquals(1.2f, ((TestService) fixture.service).repetitionPenalty);
        Thread.sleep(200);
        verify(fixture, view -> assertEquals("A short answer",
                script(view, "document.querySelector('.assistant .message-content').textContent")));
    }

    @Test
    void alternativesKeepTheOriginalAndChangeOneSettingAtATime() throws Exception {
        AlternativeService service = new AlternativeService();
        Fixture fixture = open(service);
        verifyWithoutClosing(fixture, view -> script(view,
                "setModel('stub.dj'); chatBridge.send('Hello', 80, 0.8, 40, 1234, 1)"));
        assertTrue(service.first.await(10, TimeUnit.SECONDS));
        Thread.sleep(200);
        verifyWithoutClosing(fixture, view -> script(view,
                "document.querySelector('.alternatives-button').click()"));
        assertTrue(service.extra.await(10, TimeUnit.SECONDS));
        Thread.sleep(200);
        verify(fixture, ChatWindowTest::assertAlternativesVisible);
        assertEquals(List.of(new Invocation("Hello", 80, 0.8f, 40, 1234, 1.0f),
                new Invocation("Hello", 80, 0.8f, 40, 1234, 1.1f),
                new Invocation("Hello", 80, 0.8f, 40, 1235, 1.0f)), service.calls);
    }

    private static void assertAlternativesVisible(WebView view) {
        assertEquals("response-1", script(view,
                "document.querySelector('.message.assistant > .message-content').textContent"));
        assertEquals(2, ((Number) script(view, "document.querySelectorAll('.alternative').length")).intValue());
        assertEquals("response-2,response-3", script(view,
                "[...document.querySelectorAll('.alternative .message-content')]"
                        + ".map(node => node.textContent).join(',')"));
        assertEquals(1, ((Number) script(view, "document.querySelectorAll('.message.user').length")).intValue());
        assertEquals(3, copyCount(view));
        script(view, "document.querySelector('.alternatives-button').click()");
        assertEquals(true, script(view, "document.querySelector('.alternatives').hidden"));
        script(view, "document.querySelector('.alternatives-button').click()");
        assertEquals(2, ((Number) script(view, "document.querySelectorAll('.alternative').length")).intValue());
    }

    @Test
    void newChatStopsGenerationAndAcceptsAnotherPrompt() throws Exception {
        BlockingService service = new BlockingService();
        Fixture fixture = open(service);
        verifyWithoutClosing(fixture, view -> script(view, "setModel('stub.dj'); chatBridge.send('old', 10, 1, 1, 1)"));
        assertTrue(service.started.await(10, TimeUnit.SECONDS));
        verifyWithoutClosing(fixture, view -> {
            assertEquals("Stop generating", script(view, "document.getElementById('send').title"));
            script(view, "document.getElementById('newChat').click()");
            assertEquals(false, script(view, "document.getElementById('prompt').disabled"));
            script(view, "chatBridge.send('new', 10, 1, 1, 1)");
        });
        assertTrue(service.stopped.await(10, TimeUnit.SECONDS));
        assertTrue(service.finished.await(10, TimeUnit.SECONDS));
        Thread.sleep(200);
        verify(fixture, ChatWindowTest::assertNewChat);
    }

    private static void assertNewChat(WebView view) {
        assertEquals("new", script(view, "document.querySelector('.user .message-content').textContent"));
        assertEquals("new response", script(view,
                "document.querySelector('.assistant .message-content').textContent"));
        assertEquals(1, ((Number) script(view, "document.querySelectorAll('.user').length")).intValue());
    }

    @Test
    void stopButtonCancelsGeneration() throws Exception {
        BlockingService service = new BlockingService();
        Fixture fixture = open(service);
        verifyWithoutClosing(fixture, view -> script(view, "setModel('stub.dj'); chatBridge.send('old', 10, 1, 1, 1)"));
        assertTrue(service.started.await(10, TimeUnit.SECONDS));
        verify(fixture, view -> {
            script(view, "document.getElementById('send').click()");
            assertEquals("Send message", script(view, "document.getElementById('send').title"));
            assertEquals(false, script(view, "document.getElementById('prompt').disabled"));
        });
        assertTrue(service.stopped.await(10, TimeUnit.SECONDS));
    }

    private static void pressEnter(WebView view, boolean shift) {
        Event.fireEvent(view, new KeyEvent(KeyEvent.KEY_PRESSED, "\r", "\r", KeyCode.ENTER,
                shift, false, false, false));
    }

    private static void assertSelection(WebView view) {
        String selected = (String) script(view, """
                var text = document.querySelector('.assistant .message-content');
                var range = document.createRange();
                range.selectNodeContents(text);
                window.getSelection().removeAllRanges();
                window.getSelection().addRange(range);
                window.getSelection().toString();
                """);
        assertEquals("A selectable answer ".repeat(45), selected);
    }

    private static Fixture open() throws Exception {
        return open(new TestService());
    }

    private static Fixture open(ChatService service) throws Exception {
        Fixture fixture = new Fixture(service);
        Platform.runLater(fixture::show);
        assertTrue(fixture.loaded.await(10, TimeUnit.SECONDS));
        return fixture;
    }

    private static void verify(Fixture fixture, Check check) throws Exception {
        checkOnUi(fixture, check, true);
    }

    private static void verifyWithoutClosing(Fixture fixture, Check check) throws Exception {
        checkOnUi(fixture, check, false);
    }

    private static void checkOnUi(Fixture fixture, Check check, boolean close) throws Exception {
        CountDownLatch checked = new CountDownLatch(1);
        AtomicReference<Throwable> failure = new AtomicReference<>();
        Platform.runLater(() -> fixture.check(check, failure, checked, close));
        assertTrue(checked.await(10, TimeUnit.SECONDS));
        if (failure.get() != null) throw new AssertionError(failure.get());
    }

    private static Object script(WebView view, String source) {
        return view.getEngine().executeScript(source);
    }

    private static double styleNumber(WebView view, String selector, String property) {
        return ((Number) script(view, "parseFloat(getComputedStyle(document.querySelector('"
                + selector + "'))." + property + ")")).doubleValue();
    }

    private interface Check { void run(WebView view); }

    private static final class Fixture {
        private final CountDownLatch loaded = new CountDownLatch(1);
        private final ChatService service;
        private WebView view;
        private Stage stage;

        private Fixture(ChatService service) { this.service = service; }

        private void show() {
            view = new WebView();
            stage = new Stage();
            stage.setScene(new Scene(view, 1000, 700));
            stage.show();
            view.getEngine().getLoadWorker().stateProperty().addListener((ob, old, state) -> {
                if (state == Worker.State.SUCCEEDED) loaded.countDown();
            });
            new ChatWindow(stage, view, service).open();
        }

        private void check(Check check, AtomicReference<Throwable> failure,
                           CountDownLatch done, boolean close) {
            try {
                check.run(view);
            } catch (Throwable error) {
                failure.set(error);
            } finally {
                if (close) stage.close();
                done.countDown();
            }
        }
    }

    private static final class TestService implements ChatService {
        private final CountDownLatch generated = new CountDownLatch(1);
        private volatile float repetitionPenalty;

        @Override public void loadModel(Path modelPath) {}
        @Override public boolean isModelLoaded() { return true; }
        @Override public String getLoadedModelName() { return "stub.dj"; }

        @Override
        public String generate(String prompt, int maxTokens, float temperature, int topK,
                               long seed, Consumer<String> listener) {
            listener.accept("A short");
            generated.countDown();
            return "A short answer";
        }

        @Override
        public String generate(String prompt, int maxTokens, float temperature, int topK,
                               long seed, float penalty, Consumer<String> listener) {
            repetitionPenalty = penalty;
            return generate(prompt, maxTokens, temperature, topK, seed, listener);
        }
    }

}
