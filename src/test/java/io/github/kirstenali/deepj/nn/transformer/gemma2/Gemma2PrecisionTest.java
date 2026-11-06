package io.github.kirstenali.deepj.nn.transformer.gemma2;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.layers.transformer.qwen3.Qwen3Rotary;
import io.github.kirstenali.deepj.nn.ModelOps;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Gemma2PrecisionTest {
    private static final Path ROOT = Path.of("src/test/resources/gemma2/precision");

    @BeforeEach void backend() {
        Tensor.setBackend(Boolean.getBoolean("gemma.test.metal") ? new MetalBackend() : new CpuBackend());
    }

    @Test void bfloatRotaryMatchesTheIndependentTransformersOracle() throws Exception {
        try (var scope = TensorScope.open(); var ignored = GradMode.noGrad();
             var data = TensorWeights.open(ROOT, DType.FLOAT32)) {
            Tensor actual = new Gemma2Rotary(8, 10000).apply(data.tensor("query").to(DType.BFLOAT16), 7, 5);
            assertEquals(DType.BFLOAT16, actual.dtype());
            assertArrayEquals(data.tensor("rotated").toFloatArray(), actual.toFloatArray(), 0);
        }
    }

    @Test void bfloatScoreClippingMatchesTheIndependentTransformersOracle() throws Exception {
        try (var scope = TensorScope.open(); var ignored = GradMode.noGrad();
             var data = TensorWeights.open(ROOT, DType.FLOAT32)) {
            Tensor q = data.tensor("query").to(DType.BFLOAT16), k = data.tensor("key").to(DType.BFLOAT16);
            Tensor v = data.tensor("value").to(DType.BFLOAT16);
            Tensor output = Gemma2Attention.apply(q, k, v, data.tensor("mask"), 0.3f, 2);
            assertEquals(DType.BFLOAT16, output.dtype());
            assertArrayEquals(data.tensor("attention").toFloatArray(), output.toFloatArray(), 0);
        }
    }

    @Test void float32RetainsTheExistingRotaryAndAttentionCalculations() throws Exception {
        try (var scope = TensorScope.open(); var ignored = GradMode.noGrad();
             var data = TensorWeights.open(ROOT, DType.FLOAT32)) {
            Tensor q = data.tensor("query"), k = data.tensor("key"), v = data.tensor("value");
            Tensor mask = data.tensor("mask");
            assertArrayEquals(new Qwen3Rotary(8, 10000).apply(q, 7, 5).toFloatArray(),
                    new Gemma2Rotary(8, 10000).apply(q, 7, 5).toFloatArray(), 0);
            assertArrayEquals(ModelOps.attention(q, k, v, mask, 0.3f, 2).toFloatArray(),
                    Gemma2Attention.apply(q, k, v, mask, 0.3f, 2).toFloatArray(), 0);
        }
    }
}
