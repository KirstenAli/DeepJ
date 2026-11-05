package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.diffusion.sana.SanaSprint;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Base64;
import javax.imageio.ImageIO;

record DesktopImage(byte[] png, int width, int height, int steps, long seed) {
    static DesktopImage encode(BufferedImage image, SanaSprint.Options options) throws IOException {
        var output = new ByteArrayOutputStream();
        if (!ImageIO.write(image, "png", output)) throw new IOException("PNG writer unavailable");
        return new DesktopImage(output.toByteArray(), image.getWidth(), image.getHeight(),
                options.steps(), options.seed());
    }

    String dataUrl() { return "data:image/png;base64," + Base64.getEncoder().encodeToString(png); }
    String filename() { return "deepj-image-" + seed + ".png"; }
}
