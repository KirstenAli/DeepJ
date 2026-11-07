package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.nn.ModelOps;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class VideoInferenceTilingTest {
    private static final Path WEIGHTS = Path.of("src/test/resources/sana-video/transformer");

    @BeforeEach void backend() {
        Tensor.setBackend(Boolean.getBoolean("sanavideo.test.metal") ? new MetalBackend() : new CpuBackend());
    }

    @Test void independentQueryTilesMatchCompleteAttentionWithAndWithoutAMask() {
        try (var scope = TensorScope.open()) {
            Tensor query = random(1, 2, 8193, 8), key = random(1, 2, 11, 8), value = random(1, 2, 11, 8);
            Tensor mask = Tensor.from(new float[]{0, 0, -10000, 0, 0, 0, 0, -10000, 0, 0, 0}, 1, 1, 1, 11);
            for (Tensor bias : new Tensor[]{null, mask}) {
                float[] expected = ModelOps.attention(query, key, value, bias, 0.25f, 0).toFloatArray();
                try (var ignored = GradMode.noGrad()) {
                    assertArrayEquals(expected,
                            VideoCrossAttention.apply(query, key, value, bias, 0.25f).toFloatArray(), 2e-5f);
                }
            }
        }
    }

    @Test void framewiseFeedForwardPreservesSpatialAndTemporalMixing() throws Exception {
        try (var scope = TensorScope.open(); var weights = TensorWeights.open(WEIGHTS, DType.FLOAT32)) {
            Tensor input = random(1, 24, 5, 2, 2);
            float[] expected = SanaVideoFeedForward.apply(input, weights, "transformer_blocks.0.ff").toFloatArray();
            try (var ignored = GradMode.noGrad()) {
                assertArrayEquals(expected, SanaVideoFeedForward.apply(input, weights,
                        "transformer_blocks.0.ff").toFloatArray(), 2e-5f);
            }
        }
    }

    @ParameterizedTest @EnumSource(DType.class)
    void stagedSelfAttentionPreservesTheCompleteRotaryCalculation(DType dtype) throws Exception {
        try (var scope = TensorScope.open(); var weights = TensorWeights.open(WEIGHTS, dtype)) {
            var rotary = new VideoRotary(1, 91, 91, 12);
            var attention = new SanaVideoAttention(weights, 2);
            Tensor input = random(1, 91 * 91, 24).to(dtype);
            float[] expected = attention.self(input, "transformer_blocks.0.attn1", rotary).toFloatArray();
            try (var ignored = GradMode.noGrad()) {
                assertArrayEquals(expected, attention.self(input, "transformer_blocks.0.attn1", rotary)
                        .toFloatArray(), dtype == DType.FLOAT32 ? 2e-4f : .01f);
            }
        }
    }

    @Test void longInferencePreservesResidualsAcrossAllBlockCleanupBoundaries() throws Exception {
        try (var scope = TensorScope.open(); var weights = TensorWeights.open(WEIGHTS, DType.FLOAT32)) {
            var model = new SanaVideoTransformer(WEIGHTS.resolve("config.json"), weights);
            Tensor latent = random(1, 4, 5, 82, 82), caption = random(1, 11, 16);
            int[] mask = {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0};
            float[] expected = model.forward(latent, caption, mask, 700).toFloatArray();
            try (var ignored = GradMode.noGrad()) {
                assertArrayEquals(expected, model.forward(latent, caption, mask, 700).toFloatArray(), 0.001f);
            }
        }
    }

    private static Tensor random(int... shape) {
        int count = java.util.Arrays.stream(shape).reduce(1, Math::multiplyExact);
        float[] values = new float[count];
        var random = new Random(431);
        for (int index = 0; index < count; index++) { values[index] = random.nextFloat() - 0.5f; }
        return Tensor.from(values, shape);
    }
}
