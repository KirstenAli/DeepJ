package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.nn.diffusion.sana.SanaSprint;
import io.github.kirstenali.deepj.nn.diffusion.sanavideo.SanaVideo;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import java.awt.image.BufferedImage;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class CompactMediaPipelineTest {
    @AfterEach void release() {
        Tensor.backend().releaseResources();
        Tensor.setBackend(new CpuBackend());
    }

    @ParameterizedTest @ValueSource(strings = {"cpu", "metal"})
    void compactImagePipelinesUseFloat32SamplingAndSelectedDecoderPrecision(String name) throws Exception {
        backend(name);
        var pipeline = new SanaSprint(Path.of("src/test/resources/sana/pipeline"));
        for (DType dtype : new DType[]{DType.FLOAT16, DType.BFLOAT16}) {
            var options = new SanaSprint.Options(32, 32, 2, 4.5f, 42, dtype);
            var image = pipeline.generate("a", options);
            assertEquals(32, image.getWidth());
            assertEquals(32, image.getHeight());
        }
    }

    @ParameterizedTest @EnumSource(DType.class)
    void imageSamplingPreservesNoiseAcrossDenoiserCleanup(DType dtype) throws Exception {
        Assumptions.assumeTrue(MetalBackend.isAvailable());
        var pipeline = new SanaSprint(Path.of("src/test/resources/sana/pipeline"));
        var options = new SanaSprint.Options(32, 32, 2, 4.5f, 42, dtype);
        Tensor.setBackend(new CpuBackend());
        var expected = pipeline.generate("a", options);
        Tensor.backend().releaseResources();
        Tensor.setBackend(new MetalBackend());
        closePixels(expected, pipeline.generate("a", options));
    }

    private static void closePixels(BufferedImage expected, BufferedImage actual) {
        for (int y = 0; y < expected.getHeight(); y++) {
            for (int x = 0; x < expected.getWidth(); x++) {
                for (int shift = 0; shift <= 16; shift += 8) {
                    int first = expected.getRGB(x, y) >> shift & 255, second = actual.getRGB(x, y) >> shift & 255;
                    assertTrue(Math.abs(first - second) <= 2, "CPU/Metal sampler pixel differs at " + x + "," + y);
                }
            }
        }
    }

    @ParameterizedTest @ValueSource(strings = {"cpu", "metal"})
    void compactVideoDenoisersPassThroughFloat32SamplingAndFrameDecoding(String name) throws Exception {
        backend(name);
        var pipeline = new SanaVideo(Path.of("src/test/resources/sana-video/pipeline"));
        for (DType dtype : new DType[]{DType.FLOAT16, DType.BFLOAT16}) {
            var options = new SanaVideo.Options(16, 16, 5, 2, 6, 42, dtype);
            var frames = pipeline.generate("a", options);
            assertEquals(5, frames.size());
            assertEquals(16, frames.get(0).getWidth());
            assertEquals(16, frames.get(4).getHeight());
        }
    }

    private static void backend(String name) {
        if (name.equals("metal")) Assumptions.assumeTrue(MetalBackend.isAvailable());
        Tensor.setBackend(name.equals("metal") ? new MetalBackend() : new CpuBackend());
    }
}
