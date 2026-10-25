package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.chatui.ChatWindowTestSupport.TestService;
import javafx.application.Platform;
import javafx.concurrent.Worker;
import javafx.scene.Scene;
import javafx.scene.web.WebView;
import javafx.stage.Stage;
import org.junit.jupiter.api.BeforeAll;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertTrue;

abstract class ChatDesktopTestSupport {
    private static final AtomicBoolean STARTED = new AtomicBoolean();

    @BeforeAll
    static void startJavaFx() {
        if (STARTED.compareAndSet(false, true)) {
            Platform.startup(() -> {});
            Platform.setImplicitExit(false);
        }
    }

    protected static Fixture open() throws Exception {
        return open(new TestService());
    }

    protected static Fixture open(ChatService service) throws Exception {
        Fixture fixture = new Fixture(service);
        Platform.runLater(fixture::show);
        assertTrue(fixture.loaded.await(10, TimeUnit.SECONDS));
        return fixture;
    }

    protected static void verify(Fixture fixture, Check check) throws Exception {
        checkOnUi(fixture, check, true);
    }

    protected static void verifyWithoutClosing(Fixture fixture, Check check) throws Exception {
        checkOnUi(fixture, check, false);
    }

    protected static void checkOnUi(Fixture fixture, Check check, boolean close) throws Exception {
        CountDownLatch checked = new CountDownLatch(1);
        AtomicReference<Throwable> failure = new AtomicReference<>();
        Platform.runLater(() -> fixture.check(check, failure, checked, close));
        assertTrue(checked.await(10, TimeUnit.SECONDS));
        if (failure.get() != null) throw new AssertionError(failure.get());
    }

    protected static void awaitScript(Fixture fixture, String condition) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        AtomicReference<Boolean> ready = new AtomicReference<>(false);
        while (System.nanoTime() < deadline) {
            verifyWithoutClosing(fixture, view -> ready.set(Boolean.TRUE.equals(script(view, condition))));
            if (ready.get()) return;
            Thread.sleep(20);
        }
        throw new AssertionError("Timed out waiting for desktop state: " + condition);
    }

    protected static Object script(WebView view, String source) {
        return view.getEngine().executeScript(source);
    }

    protected static double styleNumber(WebView view, String selector, String property) {
        return ((Number) script(view, "parseFloat(getComputedStyle(document.querySelector('"
                + selector + "'))." + property + ")")).doubleValue();
    }

    protected interface Check { void run(WebView view); }

    protected static final class Fixture {
        private final CountDownLatch loaded = new CountDownLatch(1);
        final ChatService service;
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

}
