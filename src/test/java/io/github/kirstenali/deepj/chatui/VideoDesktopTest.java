package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.diffusion.sanavideo.SanaVideo;
import javafx.application.Platform;
import javafx.stage.Stage;
import netscape.javascript.JSObject;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

@EnabledIfSystemProperty(named = "deepj.uiTests", matches = "true")
class VideoDesktopTest extends ChatDesktopTestSupport {
    @TempDir Path temporary;
    private static final String SEEK = "document.querySelector('.video-seek')";
    private static final String SETTINGS = "{\"width\":128,\"height\":96,\"frames\":9,\"steps\":1,"
            + "\"seed\":42,\"fps\":4,\"guidance\":6}";

    @Test void videoActuallyPlaysSeeksAndSavesThroughTheDesktopBridge() throws Exception {
        var fixture = open();
        try (var state = attach(fixture)) {
            Platform.runLater(() -> state.bridge.load(temporary));
            awaitScript(fixture, "!element('videoSettings').hidden && !element('prompt').disabled");
            verifyWithoutClosing(fixture, view ->
                    script(view, "videoBridge.generate('Motion test','" + SETTINGS + "')"));
            awaitScript(fixture, "document.querySelector('.video-frame')?.naturalWidth === 128");
            verifyPlayback(fixture);
            verifyWithoutClosing(fixture, view -> script(view, "document.querySelector('.video-save').click()"));
            awaitScript(fixture, "element('toast').textContent === 'Video saved'");
            assertTrue(Files.size(temporary.resolve("saved.mp4")) > 0);
            reviewThemes(fixture);
            verifyWithoutClosing(fixture, view -> assertTrue(styleNumber(view, ".video-save", "height") <= 30));
        }
    }

    @Test void aRealModelWorkerReachesDesktopPlaybackAndTheSavePicker() throws Exception {
        var fixture = open();
        try (var state = attach(fixture, new SanaVideoService())) {
            Path model = Path.of("src/test/resources/sana-video/pipeline").toAbsolutePath();
            Platform.runLater(() -> state.bridge.load(model));
            awaitScript(fixture, "!element('videoSettings').hidden && !element('prompt').disabled");
            String settings = "{\"width\":16,\"height\":16,\"frames\":5,\"steps\":1,"
                    + "\"seed\":42,\"fps\":4,\"guidance\":6}";
            verifyWithoutClosing(fixture, view -> script(view, "videoBridge.generate('A fox','" + settings + "')"));
            awaitScript(fixture, "document.querySelector('.video-frame')?.naturalWidth === 16");
            verifyPlayback(fixture);
            verifyWithoutClosing(fixture, view -> script(view, "document.querySelector('.video-save').click()"));
            awaitScript(fixture, "element('toast').textContent === 'Video saved'");
            assertTrue(Files.size(temporary.resolve("saved.mp4")) > 0);
        }
    }

    private static void reviewThemes(Fixture fixture) throws Exception {
        for (String theme : List.of("light", "dark", "warm", "ocean")) {
            verifyWithoutClosing(fixture, view -> script(view, "setTheme('" + theme + "')"));
            ImageDesktopSnapshots.capture(fixture, "video-player-" + theme);
            verifyWithoutClosing(fixture, view -> script(view, "element('videoSettingsButton').click()"));
            ImageDesktopSnapshots.capture(fixture, "video-settings-" + theme);
            verifyWithoutClosing(fixture, view -> script(view, "closeSettings()"));
        }
    }

    @Test void stoppingKeepsTheComposerBusyUntilVideoResourcesHaveBeenReleased() throws Exception {
        var fixture = open();
        var generator = new BlockingService();
        try (var state = attach(fixture, generator)) {
            Platform.runLater(() -> state.bridge.load(temporary));
            awaitScript(fixture, "!element('videoSettings').hidden && !element('prompt').disabled");
            verifyWithoutClosing(fixture, view -> script(view, "videoBridge.generate('Cancel','" + SETTINGS + "')"));
            assertTrue(generator.started.await(5, TimeUnit.SECONDS));
            verifyWithoutClosing(fixture, view -> script(view, "videoBridge.stop()"));
            assertTrue(generator.interrupted.await(5, TimeUnit.SECONDS));
            verifyWithoutClosing(fixture, view -> assertEquals(true, script(view, "element('prompt').disabled")));
            generator.cleanup.countDown();
            awaitScript(fixture, "!element('prompt').disabled");
            verifyWithoutClosing(fixture, view -> assertEquals(null, script(view,
                    "document.querySelector('.video-player')")));
        } finally { generator.cleanup.countDown(); }
    }

    private static void verifyPlayback(Fixture fixture) throws Exception {
        verifyWithoutClosing(fixture, view -> script(view, "document.querySelector('.video-play').click()"));
        awaitScript(fixture, SEEK + ".value > 0");
        verifyWithoutClosing(fixture, view -> {
            script(view, "document.querySelector('.video-play').click()");
            assertEquals("Play video",
                    script(view, "document.querySelector('.video-play').getAttribute('aria-label')"));
            script(view, SEEK + ".value=4; " + SEEK + ".dispatchEvent(new Event('input'))");
            assertEquals("1.00 seconds", script(view, SEEK + ".getAttribute('aria-valuetext')"));
            assertTrue(script(view, "document.querySelector('.video-frame').src").toString().endsWith("frame-4.png"));
        });
    }

    private State attach(Fixture fixture) throws Exception {
        return attach(fixture, new PreviewService());
    }

    private State attach(Fixture fixture, VideoService generator) throws Exception {
        var state = new State();
        verifyWithoutClosing(fixture, view -> {
            var window = (JSObject) view.getEngine().executeScript("window");
            state.stage = (Stage) view.getScene().getWindow();
            var events = new ImageWorkspace.Events(state.busy::get, value -> {
                state.busy.set(value);
                window.call("setBusy", value);
            }, () -> {}, window::call);
            state.bridge = new VideoWorkspace(state.stage, state.worker, fixture.service, events,
                    generator,
                    new VideoDownloads((picker, owner) -> temporary.resolve("saved.mp4").toFile()));
            window.setMember("videoBridge", state.bridge);
        });
        return state;
    }

    private static final class BlockingService implements VideoService {
        final CountDownLatch started = new CountDownLatch(1), interrupted = new CountDownLatch(1);
        final CountDownLatch cleanup = new CountDownLatch(1);
        @Override public Path resolve(Path path) { return path; }
        @Override public List<BufferedImage> generate(Path path, String prompt, SanaVideo.Options options,
                                                       Consumer<String> progress) throws InterruptedException {
            started.countDown();
            try { cleanup.await(); }
            catch (InterruptedException stopped) {
                interrupted.countDown();
                cleanup.await();
                throw stopped;
            }
            return List.of();
        }
    }

    private static final class PreviewService implements VideoService {
        @Override public Path resolve(Path path) { return path; }
        @Override public List<BufferedImage> generate(Path path, String prompt, SanaVideo.Options options,
                                                       Consumer<String> progress) {
            progress.accept("Video denoising step 1/1");
            return DesktopVideoTest.frames();
        }
    }

    private static final class State implements AutoCloseable {
        final java.util.concurrent.ExecutorService worker = Executors.newSingleThreadExecutor();
        final AtomicBoolean busy = new AtomicBoolean();
        VideoWorkspace bridge;
        Stage stage;
        @Override public void close() {
            worker.shutdownNow();
            Platform.runLater(stage::close);
        }
    }
}
