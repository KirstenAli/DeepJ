package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.diffusion.sana.SanaSprint;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.io.ByteArrayInputStream;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class DesktopImageTest {
    @Test void encodedPreviewAndDownloadAreTheSamePng() throws Exception {
        var pixels = new BufferedImage(2, 3, BufferedImage.TYPE_INT_RGB);
        pixels.setRGB(1, 1, 0x123456);
        var image = DesktopImage.encode(pixels, new SanaSprint.Options(32, 32, 2, 4.5f, 99));
        var preview = java.util.Base64.getDecoder().decode(image.dataUrl().split(",", 2)[1]);
        assertArrayEquals(image.png(), preview);
        var restored = ImageIO.read(new ByteArrayInputStream(preview));
        assertEquals(0x123456, restored.getRGB(1, 1) & 0xffffff);
        assertEquals("deepj-image-99.png", image.filename());
        assertEquals(2, image.width());
        assertEquals(3, image.height());
    }
}
