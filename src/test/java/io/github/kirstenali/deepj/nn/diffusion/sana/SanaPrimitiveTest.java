package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.nn.ModelOps;
import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.nn.GroupedConv2D;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.util.Random;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class SanaPrimitiveTest {
    @Test void metalConvolutionLayoutsMatchCpu() {
        assumeTrue(Boolean.getBoolean("sana.test.metal"));
        for (int[] config : new int[][]{{1, 4, 6, 1}, {4, 4, 4, 3}, {2, 4, 6, 3}, {3, 6, 9, 1}}) {
            Tensor.setBackend(new CpuBackend());
            float[] expected = convolution(config);
            Tensor.setBackend(new MetalBackend());
            assertArrayEquals(expected, convolution(config), 2e-5f, java.util.Arrays.toString(config));
        }
    }

    @Test void metalLinearAttentionMatchesCpu() {
        assumeTrue(Boolean.getBoolean("sana.test.metal"));
        Tensor.setBackend(new CpuBackend());
        float[] expected = attention();
        Tensor.setBackend(new MetalBackend());
        assertArrayEquals(expected, attention(), 2e-5f);
    }

    private static float[] convolution(int[] config) {
        try (var ignored = GradMode.noGrad()) {
            Tensor input = random(1, config[1], 4, 5);
            Tensor weight = random(config[2], config[1] / config[0], config[3], config[3]);
            return GroupedConv2D.apply(input, weight, null, config[0]).toFloatArray();
        }
    }

    private static float[] attention() {
        try (var ignored = GradMode.noGrad()) {
            return ModelOps.linearAttention(random(1, 3, 16, 8), random(1, 3, 16, 8),
                    random(1, 3, 16, 8), 1e-15f).toFloatArray();
        }
    }

    private static Tensor random(int... shape) {
        int count = 1;
        for (int size : shape) { count *= size; }
        Random random = new Random(192);
        float[] values = new float[count];
        for (int i = 0; i < count; i++) { values[i] = random.nextFloat() - 0.5f; }
        return Tensor.from(values, shape);
    }
}
