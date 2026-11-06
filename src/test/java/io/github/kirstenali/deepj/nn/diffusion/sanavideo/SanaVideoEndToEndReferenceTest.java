package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SanaVideoEndToEndReferenceTest {
    private static final Path ROOT = Path.of("src/test/resources/sana-video");

    @BeforeEach void backend() {
        Tensor.setBackend(Boolean.getBoolean("sanavideo.test.metal") ? new MetalBackend() : new CpuBackend());
    }

    @Test void completeFloat32PipelineMatchesPythonFromTheSameTokensAndNoise() throws Exception {
        compare();
    }

    private static void compare() throws Exception {
        List<BufferedImage> expected;
        try (var fixture = TensorWeights.open(ROOT.resolve("pipeline-reference"), DType.FLOAT32)) {
            expected = VideoFrames.rgb(fixture.tensor("float32.pixels"));
        }
        var options = new SanaVideo.Options(16, 16, 9, 5, 6, 123);
        List<BufferedImage> actual = new SanaVideo(ROOT.resolve("pipeline")).generate("a fox", options);
        assertEquals(expected.size(), actual.size());
        long total = 0;
        int maximum = 0;
        for (int frame = 0; frame < expected.size(); frame++) {
            int[] error = errors(actual.get(frame), expected.get(frame));
            total += error[0];
            maximum = Math.max(maximum, error[1]);
        }
        withinTolerance(total, maximum);
    }

    private static void withinTolerance(long total, int maximum) {
        double mean = total / (9.0 * 16 * 16 * 3);
        System.out.printf(java.util.Locale.ROOT, "PIPELINE FLOAT32 RGB max=%d mean=%.9g%n", maximum, mean);
        int maxLimit = 2;
        double meanLimit = 0.01;
        assertTrue(maximum <= maxLimit && mean <= meanLimit, "FLOAT32 RGB max=" + maximum + ", mean=" + mean);
    }

    private static int[] errors(BufferedImage actual, BufferedImage expected) {
        assertEquals(expected.getWidth(), actual.getWidth());
        assertEquals(expected.getHeight(), actual.getHeight());
        int total = 0, maximum = 0;
        for (int y = 0; y < expected.getHeight(); y++) {
            for (int x = 0; x < expected.getWidth(); x++) {
                for (int shift : new int[]{0, 8, 16}) {
                    int error = Math.abs(((actual.getRGB(x, y) >> shift) & 255)
                            - ((expected.getRGB(x, y) >> shift) & 255));
                    total += error;
                    maximum = Math.max(maximum, error);
                }
            }
        }
        return new int[]{total, maximum};
    }
}
