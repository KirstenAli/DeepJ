package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import java.nio.file.Files;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class VideoWeightCacheTest {
    private static final Path ROOT = Path.of("src/test/resources/sana-video");

    @BeforeEach void backend() {
        Tensor.setBackend(Boolean.getBoolean("sanavideo.test.metal") ? new MetalBackend() : new CpuBackend());
    }

    @Test void cachingAndStreamingProduceTheSameFloat32Predictions() throws Exception {
        try (var ignored = GradMode.noGrad(); var inputs = TensorWeights.open(ROOT, DType.FLOAT32)) {
            float[] cached = prediction(inputs, Long.MAX_VALUE);
            assertArrayEquals(cached, prediction(inputs, 0), 0);
            assertArrayEquals(inputs.tensor("transformer.output").toFloatArray(), cached, 8e-5f);
        }
    }

    private static float[] prediction(TensorWeights inputs, long budget) throws Exception {
        try (var model = SanaVideoTransformer.load(ROOT.resolve("transformer"), budget)) {
            return model.forward(inputs.tensor("transformer.input"),
                    inputs.tensor("transformer.text"), new int[]{1, 1, 1, 1, 1, 0, 0}, 733).toFloatArray();
        }
    }

    @Test void cachedSelectionsPreserveOriginalFloat32Values() throws Exception {
        String name = "patch_embedding.weight";
        try (var ignored = GradMode.noGrad(); var reference = TensorWeights.open(ROOT.resolve("transformer"),
                DType.FLOAT32); var cache = new VideoWeightCache(ROOT.resolve("transformer"), Long.MAX_VALUE)) {
            assertTrue(cache.cached());
            float[] original = reference.tensor(name).toFloatArray();
            for (int read = 0; read < 2; read++) {
                try (var weights = cache.select(name::equals)) {
                    assertArrayEquals(original, weights.tensor(name).toFloatArray(), 0);
                }
            }
        }
    }

    @Test void compactCheckpointStorageLoadsAsFloat32WorkingWeights(@TempDir Path directory) throws Exception {
        String header = "{\"weight\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}}";
        byte[] metadata = header.getBytes(StandardCharsets.UTF_8);
        byte[] bytes = ByteBuffer.allocate(12 + metadata.length).order(ByteOrder.LITTLE_ENDIAN)
                .putLong(metadata.length).put(metadata).putShort((short) 0x3fc0).putShort((short) 0xc000).array();
        Files.write(directory.resolve("model.safetensors"), bytes);
        try (var ignored = GradMode.noGrad(); var cache = new VideoWeightCache(directory, Long.MAX_VALUE);
             var weights = cache.select(name -> true)) {
            assertEquals(DType.FLOAT32, weights.tensor("weight").dtype());
            assertArrayEquals(new float[]{1.5f, -2}, weights.tensor("weight").toFloatArray(), 0);
        }
        assertArrayEquals(bytes, Files.readAllBytes(directory.resolve("model.safetensors")));
    }

    @Test void zeroBudgetStreamsAndClosedCachesCannotBeReopenedAccidentally() throws Exception {
        var cache = new VideoWeightCache(ROOT.resolve("transformer"), 0);
        assertFalse(cache.cached());
        cache.close();
        cache.close();
        assertThrows(IllegalStateException.class, () -> cache.select(name -> true));
        assertThrows(IllegalArgumentException.class, () -> new VideoWeightCache(ROOT.resolve("transformer"), -1));
        assertThrows(IllegalArgumentException.class,
                () -> SanaVideoTransformer.load(ROOT.resolve("transformer"), -1));
    }
}
