package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.tensor.Tensor;
import java.awt.image.BufferedImage;

final class SanaImages {
    private SanaImages() {}

    static BufferedImage rgb(Tensor pixels) {
        if (pixels.rank() != 4 || pixels.size(0) != 1 || pixels.size(1) != 3) {
            throw new IllegalArgumentException("Expected one RGB image");
        }
        int height = pixels.size(2), width = pixels.size(3), plane = width * height;
        float[] data = pixels.toFloatArray();
        var image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int position = 0; position < plane; position++) {
            int rgb = channel(data[position]) << 16 | channel(data[plane + position]) << 8
                    | channel(data[2 * plane + position]);
            image.setRGB(position % width, position / width, rgb);
        }
        return image;
    }

    private static int channel(float value) {
        if (!Float.isFinite(value)) throw new IllegalStateException("Non-finite generated pixel");
        return Math.round(Math.max(0, Math.min(1, (value + 1) / 2)) * 255);
    }
}
