package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.tensor.Tensor;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/** Converts one normalized RGB video tensor to Java images after network execution completes. */
public final class VideoFrames {
    private VideoFrames() {}

    public static List<BufferedImage> rgb(Tensor pixels) {
        if (pixels.rank() != 5 || pixels.size(0) != 1 || pixels.size(1) != 3) {
            throw new IllegalArgumentException("Expected one RGB video [1,3,time,height,width]");
        }
        float[] values = pixels.toFloatArray();
        int plane = pixels.size(3) * pixels.size(4), channel = pixels.size(2) * plane;
        List<BufferedImage> frames = new ArrayList<>();
        for (int time = 0; time < pixels.size(2); time++) {
            var image = new BufferedImage(pixels.size(4), pixels.size(3), BufferedImage.TYPE_INT_RGB);
            for (int point = 0; point < plane; point++) {
                int offset = time * plane + point;
                int rgb = component(values[offset]) << 16 | component(values[channel + offset]) << 8
                        | component(values[2 * channel + offset]);
                image.setRGB(point % pixels.size(4), point / pixels.size(4), rgb);
            }
            frames.add(image);
        }
        return List.copyOf(frames);
    }

    private static int component(float value) {
        if (!Float.isFinite(value)) throw new IllegalStateException("Non-finite generated video pixel");
        return (int) Math.rint((Math.max(-1, Math.min(1, value)) + 1) * 127.5f);
    }
}
