package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.nn.diffusion.sana.SanaSprint;
import io.github.kirstenali.deepj.nn.diffusion.sanavideo.SanaVideo;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
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
