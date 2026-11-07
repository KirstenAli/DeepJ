package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class VideoPrecisionTest {
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
    void zeroLinearAttentionHeadsRemainFiniteEvenWhenHalfCannotRepresentTheEpsilon(DType dtype) throws Exception {
        try (var ignored = GradMode.noGrad(); var fixture = TensorWeights.open(ROOT, dtype);
             var weights = TensorWeights.open(ROOT.resolve("transformer"), dtype)) {
            for (String name : weights.descriptors().keySet()) {
                if (name.contains(".attn1.norm_")) weights.tensor(name).multiplyScalarInPlace(0);
            }
            var model = new SanaVideoTransformer(ROOT.resolve("transformer/config.json"), weights);
            Tensor output = model.forward(fixture.tensor("transformer.input"), fixture.tensor("transformer.text"),
                    new int[]{1, 1, 1, 1, 1, 0, 0}, 733);
            for (float value : output.toFloatArray()) { assertTrue(Float.isFinite(value)); }
            assertEquals(dtype, output.dtype());
        }
    }

    @ParameterizedTest @EnumSource(DType.class)
    void normalizationReceivesFiniteProjectionsWhenTheUnnormalizedValuesExceedHalfRange(DType dtype) {
        Tensor matrix = Tensor.ones(dtype, 8, 8), scale = Tensor.ones(dtype, 8);
        try (var ignored = GradMode.noGrad(); var weights = TensorWeights.from(Map.of(
                "attention.to_q.weight", matrix.multiplyScalar(100), "attention.norm_q.weight", scale,
                "attention.to_k.weight", matrix, "attention.norm_k.weight", scale,
                "attention.to_v.weight", matrix, "attention.to_out.0.weight", matrix))) {
            Tensor input = Tensor.ones(dtype, 1, 3, 8).multiplyScalar(500);
            Tensor caption = Tensor.ones(dtype, 1, 5, 8);
            Tensor output = new SanaVideoAttention(weights, 2).cross(input, caption, null, "attention");
            assertEquals(dtype, output.dtype());
            for (float value : output.toFloatArray()) { assertEquals(64, value, 0.001); }
        }
    }

    @Test void halfGatedResidualsAccumulateInFloat32BeforeTheirProductCanOverflow() {
        Tensor input = Tensor.from(new float[]{60000}, DType.FLOAT16, 1);
        Tensor update = Tensor.from(new float[]{1000}, DType.FLOAT16, 1);
        Tensor gate = Tensor.from(new float[]{100}, DType.FLOAT16, 1);
        Tensor result = SanaVideoBlocks.gated(input, update, gate);
        assertEquals(DType.FLOAT32, result.dtype());
        assertEquals(160000, result.item());
    }
}
