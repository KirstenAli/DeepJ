package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class VideoBlockCacheTest {
    private static final Path ROOT = Path.of("src/test/resources/sana-video");
    private static final Path MODEL = ROOT.resolve("transformer");
    private final TensorBackend previous = Tensor.backend();

    @BeforeEach void backend() {
        Tensor.setBackend(Boolean.getBoolean("sanavideo.test.metal") ? new MetalBackend() : new CpuBackend());
    }

    @AfterEach void restore() {
        Tensor.backend().releaseResources();
        Tensor.setBackend(previous);
    }

    @ParameterizedTest @EnumSource(DType.class)
    void admittedBlocksSurviveReaderCleanupAndOtherBlocksStayWithinBudget(DType dtype, @TempDir Path directory)
            throws Exception {
        Path checkpoint = directory.resolve("model.safetensors");
        Files.copy(MODEL.resolve("model.safetensors"), checkpoint);
        long budget = blockBytes(dtype);
        try (var ignored = GradMode.noGrad(); var cache = new VideoWeightCache(directory, budget, dtype)) {
            float[] expected = reference(dtype);
            prime(cache);
            Files.delete(checkpoint);
            Tensor.backend().releaseTemporaryResources();
            assertArrayEquals(expected, selected(cache));
            assertThrows(IOException.class,
                    () -> cache.selectRepeated("transformer_blocks.1."));
        }
    }

    private static float[] reference(DType dtype) throws IOException {
        try (var weights = TensorWeights.open(MODEL, dtype)) {
            return weights.tensor("transformer_blocks.0.scale_shift_table").toFloatArray();
        }
    }

    private static void prime(VideoWeightCache cache) throws IOException {
        Tensor borrowed;
        try (var selected = cache.selectRepeated("transformer_blocks.0.")) {
            borrowed = selected.weights().tensor("transformer_blocks.0.scale_shift_table");
        }
        assertTrue(borrowed.retainsDeviceBuffer(), "The cache must retain its own alias after a reader closes");
    }

    private static float[] selected(VideoWeightCache cache) throws IOException {
        try (var selected = cache.selectRepeated("transformer_blocks.0.")) {
            return selected.weights().tensor("transformer_blocks.0.scale_shift_table").toFloatArray();
        }
    }

    private static long blockBytes(DType dtype) throws IOException {
        try (var weights = TensorWeights.open(MODEL, dtype)) {
            return weights.descriptors().entrySet().stream()
                    .filter(entry -> entry.getKey().startsWith("transformer_blocks.0."))
                    .mapToLong(entry -> dtype.storageBytes(entry.getValue().elements())).sum();
        }
    }

    @ParameterizedTest @EnumSource(DType.class)
    void partialCachingPreservesRepeatedPredictionsExactly(DType dtype) throws Exception {
        try (var ignored = GradMode.noGrad(); var inputs = TensorWeights.open(ROOT, dtype)) {
            float[] expected = predictions(inputs, dtype, 0);
            assertArrayEquals(expected, predictions(inputs, dtype, blockBytes(dtype)), 0);
        }
    }

    private static float[] predictions(TensorWeights inputs, DType dtype, long budget) throws IOException {
        try (var model = SanaVideoTransformer.load(MODEL, budget, dtype)) {
            Tensor latent = inputs.tensor("transformer.input"), text = inputs.tensor("transformer.text");
            int[] valid = {1, 1, 1, 1, 1, 0, 0};
            float[] first = model.forward(latent, text, valid, 733).toFloatArray();
            assertArrayEquals(first, model.forward(latent, text, valid, 733).toFloatArray(), 0);
            return first;
        }
    }
}
