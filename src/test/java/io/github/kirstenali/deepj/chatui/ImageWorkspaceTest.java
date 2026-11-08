package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.diffusion.sana.SanaSprint;
import io.github.kirstenali.deepj.tensor.DType;
import javafx.application.Platform;
import javafx.scene.web.WebView;
import netscape.javascript.JSObject;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

@EnabledIfSystemProperty(named = "deepj.uiTests", matches = "true")
class ImageWorkspaceTest extends ChatDesktopTestSupport {
    @TempDir Path temporary;

    @Test void imageBridgeLoadsAndRendersARealPngInTheExistingWindow() throws Exception {
        var fixture = open();
        Service generator = new Service(false);
        try (var state = attach(fixture, generator)) {
            state.load(temporary);
            awaitScript(fixture, "!element('imageSettings').hidden && !element('prompt').disabled");
            verifyWithoutClosing(fixture, view -> {
                assertTrue(state.service.unloaded.get());
                script(view, "element('imageDtype').value='BFLOAT16'; element('prompt').value='A fox'; submitPrompt()");
            });
            awaitScript(fixture, "document.querySelector('.image-save') !== null");
            verify(fixture, view -> {
                assertEquals(DType.BFLOAT16, generator.options.get().dtype());
                assertEquals("1024 × 1024 · 2 steps",
                        script(view, "document.querySelector('.image-meta').textContent"));
                assertEquals(false, script(view, "element('prompt').disabled"));
                assertTrue(styleNumber(view, ".generated-image", "borderRadius") >= 12);
            });
        }
    }

    @Test void cancellationWaitsForWorkerCleanupBeforeEnablingTheComposer() throws Exception {
        var fixture = open();
        Service generator = new Service(true);
        try (var state = attach(fixture, generator)) {
            state.load(temporary);
            awaitScript(fixture, "!element('imageSettings').hidden && !element('prompt').disabled");
            verifyWithoutClosing(fixture, view -> script(view, "imageBridge.generate('fox',32,32,2,1,4.5)"));
            assertTrue(generator.started.await(5, TimeUnit.SECONDS));
            verifyWithoutClosing(fixture, view -> script(view, "imageBridge.stop()"));
            assertTrue(generator.interrupted.await(5, TimeUnit.SECONDS));
            verifyWithoutClosing(fixture, view -> assertEquals(true, script(view, "element('prompt').disabled")));
            generator.cleanup.countDown();
            awaitScript(fixture, "!element('prompt').disabled");
            verify(fixture, view -> assertEquals(0, ((Number) script(view,
                    "document.querySelectorAll('.generated-image').length")).intValue()));
        }
    }

    @Test void lateImageLoadsKeepEarlierResultsVisibleAndSaveable() throws Exception {
        var fixture = open();
        try (var state = attach(fixture, new Service(false))) {
            delayImageLoads(fixture);
            state.load(temporary);
            awaitScript(fixture, "!element('imageSettings').hidden && !element('prompt').disabled");
            for (int count = 1; count <= 3; count++) {
                generateImage(fixture, count);
            }
            verify(fixture, ImageWorkspaceTest::verifyEarlierResults);
        }
    }

    private static void verifyEarlierResults(WebView view) {
        script(view, "conversation.scrollTop=0;"
                + "document.querySelector('.generated-image').dispatchEvent(new Event('contextrestored'))");
        assertEquals(0, ((Number) script(view, "conversation.scrollTop")).intValue());
        assertEquals(3, ((Number) script(view, "document.querySelectorAll('.image-save').length")).intValue());
        assertEquals(true, script(view, "[...document.querySelectorAll('.generated-image')]"
                + ".every(i=>i.dataset.ready==='true' && i.width===1024)"));
    }

    private static void generateImage(Fixture fixture, int count) throws Exception {
        verifyWithoutClosing(fixture, view -> script(view,
                "imageBridge.generate('Image " + count + "',1024,1024,2," + count + ",4.5)"));
        awaitScript(fixture, "document.querySelectorAll('.generated-image').length === " + count
                + " && !element('prompt').disabled && [...document.querySelectorAll('.generated-image')]"
                + ".every(image=>image.dataset.ready==='true')");
    }

    private static void delayImageLoads(Fixture fixture) throws Exception {
        verifyWithoutClosing(fixture, view -> script(view,
                "const create=document.createElement.bind(document);"
                + "document.createElement=tag=>{const image=create(tag);"
                + "if(tag==='img')image.addEventListener('load',event=>{"
                + "event.stopImmediatePropagation();setTimeout(()=>image.onload.call(image,event),100);});"
                + "return image;};"));
    }

    @Test void pngSavingUsesTheExactEncodedBytesOnTheSaveWorker() throws Exception {
        CountDownLatch saved = new CountDownLatch(1);
        Path target = temporary.resolve("output.png");
        var image = DesktopImage.encode(new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB),
                new SanaSprint.Options(32, 32, 1, 4.5f, 1));
        try (var downloads = new ImageDownloads()) {
            Platform.runLater(() -> downloads.save(target.toFile(), image, ignored -> saved.countDown()));
            assertTrue(saved.await(5, TimeUnit.SECONDS));
            assertArrayEquals(image.png(), java.nio.file.Files.readAllBytes(target));
        }
    }

    @Test void saveButtonReachesTheNativePngPickerAndWritesItsChosenFile() throws Exception {
        var fixture = open();
        Path destination = temporary.resolve("chosen.png");
        var downloads = new ImageDownloads((chooser, owner) -> {
            assertEquals("Save image", chooser.getTitle());
            assertEquals("deepj-image-42.png", chooser.getInitialFileName());
            assertEquals(java.util.List.of("*.png"), chooser.getExtensionFilters().get(0).getExtensions());
            return destination.toFile();
        });
        try (var state = attach(fixture, new Service(false), downloads)) {
            state.load(temporary);
            awaitScript(fixture, "!element('imageSettings').hidden && !element('prompt').disabled");
            verifyWithoutClosing(fixture, view -> script(view, "imageBridge.generate('fox',32,32,2,42,4.5)"));
            awaitScript(fixture, "document.querySelector('.image-save') !== null");
            verifyWithoutClosing(fixture, view -> script(view, "document.querySelector('.image-save').click()"));
            awaitScript(fixture, "element('toast').textContent === 'Image saved'");
            assertEquals(32, javax.imageio.ImageIO.read(destination.toFile()).getWidth());
        }
    }

    private static State attach(Fixture fixture, Service generator) throws Exception {
        return attach(fixture, generator, new ImageDownloads());
    }

    private static State attach(Fixture fixture, Service generator, ImageDownloads downloads) throws Exception {
        State state = new State();
        verifyWithoutClosing(fixture, view -> {
            var window = (JSObject) view.getEngine().executeScript("window");
            var stage = (javafx.stage.Stage) view.getScene().getWindow();
            state.stage = stage;
            var events = new ImageWorkspace.Events(state.busy::get, value -> {
                state.busy.set(value);
                window.call("setBusy", value);
            }, () -> {}, window::call);
            state.bridge.set(new ImageWorkspace(stage, state.worker, state.service, events, generator, downloads));
            window.setMember("imageBridge", state.bridge.get());
        });
        return state;
    }

    private static final class State implements AutoCloseable {
        final java.util.concurrent.ExecutorService worker = Executors.newSingleThreadExecutor();
        final AtomicBoolean busy = new AtomicBoolean();
        final AtomicReference<ImageWorkspace> bridge = new AtomicReference<>();
        final UnloadingService service = new UnloadingService();
        javafx.stage.Stage stage;
        void load(Path directory) { Platform.runLater(() -> bridge.get().load(directory)); }
        @Override public void close() {
            worker.shutdownNow();
            Platform.runLater(stage::close);
        }
    }

    private static final class UnloadingService implements ChatService {
        final AtomicBoolean unloaded = new AtomicBoolean();
        @Override public void loadModel(Path path) {}
        @Override public void unloadModel() { unloaded.set(true); }
        @Override public boolean isModelLoaded() { return !unloaded.get(); }
        @Override public String getLoadedModelName() { return "test"; }
        @Override public String generate(String prompt, int tokens, float temperature, int topK,
                                         long seed, Consumer<String> progress) { return "unused"; }
    }

    private static final class Service implements ImageService {
        final CountDownLatch started = new CountDownLatch(1), interrupted = new CountDownLatch(1);
        final CountDownLatch cleanup = new CountDownLatch(1);
        private final boolean blocking;
        final AtomicReference<SanaSprint.Options> options = new AtomicReference<>();
        Service(boolean blocking) { this.blocking = blocking; }
        @Override public Path resolve(Path path) { return path; }
        @Override public BufferedImage generate(Path path, String prompt, SanaSprint.Options options,
                                                Consumer<String> progress) throws Exception {
            this.options.set(options);
            progress.accept("Denoising step 1/2");
            started.countDown();
            if (blocking) block();
            return new BufferedImage(options.width(), options.height(), BufferedImage.TYPE_INT_RGB);
        }
        private void block() throws InterruptedException {
            try { Thread.sleep(30000); }
            catch (InterruptedException expected) {
                interrupted.countDown();
                cleanup.await();
                throw expected;
            }
        }
    }
}
