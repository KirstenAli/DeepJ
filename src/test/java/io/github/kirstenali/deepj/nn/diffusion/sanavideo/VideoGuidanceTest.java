package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class VideoGuidanceTest {
    private static final Path ROOT = Path.of("src/test/resources/sana-video");

    private final TensorBackend previous = Tensor.backend();

    @BeforeEach void backend() {
        Tensor.setBackend(Boolean.getBoolean("sanavideo.test.metal") ? new MetalBackend() : new CpuBackend());
    }

    @AfterEach void restore() {
        Tensor.backend().releaseResources();
        Tensor.setBackend(previous);
    }

    @ParameterizedTest @EnumSource(DType.class)
    void batchedGuidanceMatchesIndependentPositiveAndNegativePredictions(DType dtype) throws Exception {
        try (var ignored = GradMode.noGrad(); var fixture = TensorWeights.open(ROOT, dtype);
             var model = SanaVideoTransformer.load(ROOT.resolve("transformer"), 0, dtype)) {
            Tensor latent = fixture.tensor("transformer.input"), text = fixture.tensor("transformer.text");
            compareGuidance(model, latent, text, dtype);
        }
    }

    private static Tensor snapshot(Tensor tensor) {
        return Tensor.from(tensor.toFloatArray(), tensor.dtype(), tensor.shape());
    }

    @ParameterizedTest @EnumSource(DType.class)
    void sequentialGuidancePreservesInputsAcrossStreamedPredictions(DType dtype) throws Exception {
        try (var ignored = GradMode.noGrad(); var fixture = TensorWeights.open(ROOT, dtype);
             var model = SanaVideoTransformer.load(ROOT.resolve("transformer"), 0, dtype)) {
            Tensor latent = Tensor.random(12, 112 * 112, new java.util.Random(123))
                    .reshape(1, 4, 3, 112, 112).to(dtype);
            assertTrue(latent.numel() > 65536, "Exercise sequential guidance instead of the batched branch");
            compareGuidance(model, latent, fixture.tensor("transformer.text"), dtype);
        }
    }

    private static void compareGuidance(SanaVideoTransformer model, Tensor latent, Tensor text, DType dtype) {
        Tensor negative = snapshot(text.multiplyScalar(0.5f));
        int[] valid = {1, 1, 1, 1, 1, 0, 0}, negativeValid = {1, 1, 1, 0, 0, 0, 0};
        Tensor positive = snapshot(model.forward(latent, text, valid, 733).to(DType.FLOAT32));
        Tensor other = model.forward(latent, negative, negativeValid, 733).to(DType.FLOAT32);
        float[] expected = other.add(positive.subtract(other).multiplyScalar(6)).toFloatArray();
        Tensor actual = VideoGuidance.predict(model, latent, text, negative, valid, negativeValid, 733, 6, dtype);
        assertArrayEquals(expected, actual.toFloatArray(), dtype == DType.FLOAT32 ? 1e-4f : 0.04f);
        assertArrayEquals(latent.shape(), actual.shape());
        assertEquals(DType.FLOAT32, actual.dtype());
    }

    @ParameterizedTest @EnumSource(DType.class)
    void unguidedPredictionUsesOnlyItsOwnCaption(DType dtype) throws Exception {
        try (var ignored = GradMode.noGrad(); var fixture = TensorWeights.open(ROOT, dtype);
             var model = SanaVideoTransformer.load(ROOT.resolve("transformer"), 0, dtype)) {
            Tensor latent = fixture.tensor("transformer.input"), text = fixture.tensor("transformer.text");
            int[] valid = {1, 1, 1, 1, 1, 0, 0};
            float[] expected = model.forward(latent, text, valid, 733).toFloatArray();
            Tensor output = VideoGuidance.predict(model, latent, text, text, valid, valid, 733, 1, dtype);
            assertArrayEquals(expected, output.toFloatArray());
        }
    }
}
