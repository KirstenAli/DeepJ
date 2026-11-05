package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.chatui.ChatDesktopTestSupport.Fixture;
import javafx.scene.image.Image;
import javafx.scene.image.PixelFormat;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Capture after two rendering pulses so WebKit has painted the requested UI state. */
final class ImageDesktopSnapshots {
    private ImageDesktopSnapshots() {}
    static Path output() {
        return Path.of(System.getProperty("deepj.uiSnapshotDir", "target/image-desktop-review"));
    }

    static void capture(Fixture fixture, String name) throws Exception {
        var saved = new CountDownLatch(1);
        var failure = new AtomicReference<Throwable>();
        ChatDesktopTestSupport.verifyWithoutClosing(fixture, view -> view.snapshot(first -> {
            view.snapshot(second -> {
                save(second.getImage(), name, saved, failure);
                return null;
            }, null, null);
            return null;
        }, null, null));
        assertTrue(saved.await(10, TimeUnit.SECONDS));
        if (failure.get() != null) throw new AssertionError(failure.get());
    }

    private static void save(Image image, String name, CountDownLatch saved, AtomicReference<Throwable> failure) {
        try {
            Files.createDirectories(output());
            int width = (int) image.getWidth(), height = (int) image.getHeight();
            int[] pixels = new int[width * height];
            image.getPixelReader().getPixels(0, 0, width, height, PixelFormat.getIntArgbInstance(), pixels, 0, width);
            var png = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
            png.setRGB(0, 0, width, height, pixels, 0, width);
            javax.imageio.ImageIO.write(png, "png", output().resolve(name + ".png").toFile());
        } catch (java.io.IOException error) { failure.set(error); }
        finally { saved.countDown(); }
    }
}
