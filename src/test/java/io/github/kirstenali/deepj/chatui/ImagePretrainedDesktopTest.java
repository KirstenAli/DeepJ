package io.github.kirstenali.deepj.chatui;

import javafx.scene.web.WebView;
import netscape.javascript.JSObject;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

/** Opt-in full-checkpoint test; ordinary CI uses the deterministic small model fixtures. */
@EnabledIfSystemProperty(named = "deepj.sanaUiCheckpoint", matches = ".+")
class ImagePretrainedDesktopTest extends ChatDesktopTestSupport {
    @Test void generatesPreviewsAndExportsTheOfficialCheckpointThroughTheDesktopBridge() throws Exception {
        var fixture = open(new DeepJChatService());
        try {
            verifyWithoutClosing(fixture, ImagePretrainedDesktopTest::load);
            awaitScript(fixture, "!element('imageSettings').hidden && !element('prompt').disabled");
            ImageDesktopSnapshots.capture(fixture, "ready");
            verifyWithoutClosing(fixture, view -> script(view, "openImageSettings()"));
            ImageDesktopSnapshots.capture(fixture, "settings");
            verifyWithoutClosing(fixture, this::generate);
            waitForImage(fixture);
            verifyWithoutClosing(fixture, this::checkImage);
            ImageDesktopSnapshots.capture(fixture, "image-light");
        } finally {
            verify(fixture, view -> {});
        }
    }

    private static void load(WebView view) {
        var window = (JSObject) script(view, "window");
        var bridge = (ImageWorkspace) window.getMember("imageBridge");
        bridge.load(Path.of(System.getProperty("deepj.sanaUiCheckpoint")));
    }

    private void generate(WebView view) {
        script(view, """
                closeSettings();
                element('prompt').value='A red fox in a snowy forest, soft morning light, detailed wildlife photograph';
                submitPrompt();
                """);
    }

    private static void waitForImage(Fixture fixture) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.MINUTES.toNanos(4);
        var ready = new java.util.concurrent.atomic.AtomicBoolean();
        while (!ready.get() && System.nanoTime() < deadline) {
            verifyWithoutClosing(fixture, view -> ready.set(Boolean.TRUE.equals(script(view,
                    "document.querySelector('.generated-image')?.naturalWidth === 1024"))));
            Thread.sleep(100);
        }
        assertTrue(ready.get(), "No full-checkpoint desktop image appeared");
    }

    private void checkImage(WebView view) {
        assertEquals(false, script(view, "element('prompt').disabled"));
        assertTrue(((Number) script(view, "document.querySelector('.image-save').getBoundingClientRect().bottom"
                + " <= element('composerArea').getBoundingClientRect().top ? 1 : 0")).intValue() == 1);
        String uri = (String) script(view, "document.querySelector('.generated-image').src");
        try {
            byte[] png = java.util.Base64.getDecoder().decode(uri.split(",", 2)[1]);
            Files.write(ImageDesktopSnapshots.output().resolve("desktop-image.png"), png);
            var image = javax.imageio.ImageIO.read(new java.io.ByteArrayInputStream(png));
            assertEquals(1024, image.getWidth());
            assertEquals(1024, image.getHeight());
        } catch (java.io.IOException error) { throw new AssertionError(error); }
    }

}
