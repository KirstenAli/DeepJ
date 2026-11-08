package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.diffusion.sana.SanaSprint;
import java.awt.image.BufferedImage;
import javafx.scene.web.WebView;
import netscape.javascript.JSObject;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import static org.junit.jupiter.api.Assertions.assertEquals;

@EnabledIfSystemProperty(named = "deepj.uiTests", matches = "true")
class ImagePreviewRetentionTest extends ChatDesktopTestSupport {
    private static final int[] COLORS = {0xe93f36, 0x16834b, 0x196fc7};

    @Test
    void pixelsSurviveScrollingCollectionAndPreviewContextRestoration() throws Exception {
        var fixture = open();
        try {
            verifyWithoutClosing(fixture, view -> script(view, "setImageModel('SANA-Sprint')"));
            for (int id = 0; id < COLORS.length; id++) { add(fixture, id); }
            for (int cycle = 0; cycle < 3; cycle++) {
                verifyWithoutClosing(fixture, view -> script(view, "followLatest();openSettings();closeSettings()"));
                System.gc();
                checkAll(fixture);
            }
            restore(fixture);
            checkAll(fixture);
        } finally { verify(fixture, view -> {}); }
    }

    private static void add(Fixture fixture, int id) throws Exception {
        var pixels = new BufferedImage(1024, 1024, BufferedImage.TYPE_INT_RGB);
        var graphics = pixels.createGraphics();
        graphics.setColor(new java.awt.Color(COLORS[id]));
        graphics.fillRect(0, 0, 1024, 1024);
        graphics.dispose();
        var image = DesktopImage.encode(pixels, new SanaSprint.Options(1024, 1024, 2, 4.5f, id));
        verifyWithoutClosing(fixture, view -> {
            var window = (JSObject) script(view, "window");
            window.call("startImage", "Image " + id, id);
            window.call("finishImage", id, image.dataUrl(), 1024, 1024, 2, id);
        });
        awaitScript(fixture, "element('response-image-" + id + "').querySelector('canvas').dataset.ready==='true'");
    }

    private static void checkAll(Fixture fixture) throws Exception {
        for (int id = 0; id < COLORS.length; id++) {
            int index = id;
            verifyWithoutClosing(fixture, view -> script(view,
                    "element('response-image-" + index + "').scrollIntoView({block:'center'})"));
            ImageDesktopSnapshots.capture(fixture, "retained-image-" + id);
            verifyWithoutClosing(fixture, view -> assertPainted(view, index));
        }
    }

    private static void assertPainted(WebView view, int id) {
        String canvas = "element('response-image-" + id + "').querySelector('canvas')";
        Number x = (Number) script(view, canvas + ".getBoundingClientRect().left+" + canvas + ".clientWidth/2");
        Number y = (Number) script(view, canvas + ".getBoundingClientRect().top+" + canvas + ".clientHeight/2");
        int painted = view.snapshot(null, null).getPixelReader().getArgb(x.intValue(), y.intValue());
        assertEquals(COLORS[id], painted & 0xffffff, "Displayed preview pixels for image " + id);
        assertEquals(3, ((Number) script(view, "document.querySelectorAll('.image-save').length")).intValue());
    }

    private static void restore(Fixture fixture) throws Exception {
        verifyWithoutClosing(fixture, view -> script(view,
                "document.querySelectorAll('.generated-image').forEach(canvas=>{"
                        + "canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);"
                        + "canvas.dispatchEvent(new Event('contextrestored'));})"));
    }
}
