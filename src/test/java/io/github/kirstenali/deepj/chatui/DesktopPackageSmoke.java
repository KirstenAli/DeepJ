package io.github.kirstenali.deepj.chatui;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.util.Base64;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import javafx.animation.AnimationTimer;
import javafx.application.Platform;
import javafx.concurrent.Worker;
import javafx.scene.web.WebView;
import javafx.stage.Stage;
import javax.imageio.ImageIO;
import netscape.javascript.JSObject;

/** Runs the real desktop window from the packaged JAR without downloading a model. */
public final class DesktopPackageSmoke {
    private static final CountDownLatch FINISHED = new CountDownLatch(1);
    private static final AtomicReference<Throwable> FAILURE = new AtomicReference<>();
    private static ChatApp application;

    private DesktopPackageSmoke() {}

    public static void main(String[] args) throws Exception {
        try {
            Platform.startup(DesktopPackageSmoke::open);
            require(FINISHED.await(30, TimeUnit.SECONDS), "Desktop startup timed out");
            if (FAILURE.get() != null) throw new AssertionError("Packaged desktop check failed", FAILURE.get());
            System.out.println("DESKTOP_PACKAGE_SMOKE_PASS backend=" + ChatBackend.name());
        } finally { Platform.exit(); }
    }

    private static void open() {
        try {
            application = new ChatApp();
            var stage = new Stage();
            application.start(stage);
            var view = (WebView) stage.getScene().getRoot();
            view.getEngine().getLoadWorker().stateProperty().addListener((observable, before, state) -> {
                if (state == Worker.State.SUCCEEDED) ready(view);
                if (state == Worker.State.FAILED) fail(view.getEngine().getLoadWorker().getException());
            });
        } catch (Throwable error) { fail(error); }
    }

    private static void ready(WebView view) {
        try {
            require(WebView.class.getProtectionDomain().getCodeSource().getLocation().equals(
                    ChatApp.class.getProtectionDomain().getCodeSource().getLocation()), "External JavaFX runtime used");
            require(Boolean.TRUE.equals(script(view, "!!window.chatBridge && !!window.executionBridge")),
                    "Desktop bridges did not connect");
            showPreview(view);
            awaitPreview(view);
        } catch (Throwable error) { fail(error); }
    }

    private static void showPreview(WebView view) throws Exception {
        var image = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        var graphics = image.createGraphics();
        graphics.setColor(new java.awt.Color(0x196fc7));
        graphics.fillRect(0, 0, 32, 32);
        graphics.dispose();
        var png = new ByteArrayOutputStream();
        require(ImageIO.write(image, "png", png), "PNG encoder unavailable");
        var window = (JSObject) script(view, "window");
        window.call("setImageModel", "Desktop packaging check");
        window.call("startImage", "Preview check", 1);
        window.call("finishImage", 1, "data:image/png;base64," + Base64.getEncoder().encodeToString(png.toByteArray()),
                32, 32, 1, 42);
        window.call("setGenerating", false);
    }

    private static void awaitPreview(WebView view) {
        new AnimationTimer() {
            @Override public void handle(long now) {
                if (!Boolean.TRUE.equals(script(view,
                        "document.querySelector('canvas').dataset.ready==='true'"))) return;
                stop();
                verifyOptions(view);
            }
        }.start();
    }

    private static void verifyOptions(WebView view) {
        try {
            require("25,111,199,255".equals(script(view,
                    "Array.from(document.querySelector('canvas').getContext('2d')"
                            + ".getImageData(0,0,1,1).data).join(',')")),
                    "Preview pixels changed");
            script(view, "setModel('Chat');openSettings()");
            require(Boolean.FALSE.equals(script(view, "element('executionBackend').disabled")), "Backend unavailable");
            require("2048".equals(script(view, "element('maxTokens').value")), "Generation controls did not load");
            script(view, "closeSettings();setVideoModel('Video');openSettings()");
            require(Boolean.FALSE.equals(script(view, "element('videoSettings').hidden")),
                    "Video controls unavailable");
            FINISHED.countDown();
        } catch (Throwable error) { fail(error); }
    }

    private static Object script(WebView view, String source) { return view.getEngine().executeScript(source); }
    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
    private static void fail(Throwable error) {
        FAILURE.compareAndSet(null, error);
        FINISHED.countDown();
    }
}
