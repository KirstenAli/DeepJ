package io.github.kirstenali.deepj.nn.transformer.gemma2;

import io.github.kirstenali.deepj.layers.transformer.qwen3.Qwen3RotaryTables;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class Gemma2RotaryTest {
    @BeforeEach void backend() {
        Tensor.setBackend(Boolean.getBoolean("gemma.test.metal") ? new MetalBackend() : new CpuBackend());
    }

    @ParameterizedTest @EnumSource(DType.class)
    void cachedTablesPreserveValuesAndGradientsAcrossRetiredLayerScopes(DType dtype) {
        var rotary = new Gemma2Rotary(64, 10000);
        Tensor original = values(dtype).requiresGrad(true);
        Tensor expected = rotary.apply(original, 5, 7);
        float[] output = expected.toFloatArray();
        expected.sumTensor().backward();
        float[] gradient = original.grad().toFloatArray();
        try (var tables = Qwen3RotaryTables.open()) {
            for (int layer = 0; layer < 4; layer++) { check(rotary, dtype, output, gradient); }
        }
    }

    private static void check(Gemma2Rotary rotary, DType dtype, float[] output, float[] gradient) {
        try (var scope = TensorScope.open()) {
            Tensor input = values(dtype).requiresGrad(true), actual = rotary.apply(input, 5, 7);
            assertArrayEquals(output, actual.toFloatArray());
            actual.sumTensor().backward();
            assertArrayEquals(gradient, input.grad().toFloatArray());
        }
        Tensor.backend().releaseTemporaryResources();
    }

    private static Tensor values(DType dtype) {
        float[] values = new float[2 * 5 * 64];
        Random random = new Random(23);
        for (int index = 0; index < values.length; index++) { values[index] = random.nextFloat() - .5f; }
        return Tensor.from(values, dtype, 1, 2, 5, 64);
    }
}
