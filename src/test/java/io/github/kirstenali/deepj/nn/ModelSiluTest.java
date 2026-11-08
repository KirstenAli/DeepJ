package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class ModelSiluTest {
    private final TensorBackend previous = Tensor.backend();

    @BeforeEach void backend() {
        boolean metal = Boolean.getBoolean("sana.test.metal") || Boolean.getBoolean("deepj.requireMetal");
        Tensor.setBackend(metal ? new MetalBackend() : new CpuBackend());
    }

    @AfterEach void restore() {
        Tensor.backend().releaseResources();
        Tensor.setBackend(previous);
    }

    @ParameterizedTest @EnumSource(DType.class)
    void roundsTheCompletedActivationAndKeepsFloat32Gradients(DType dtype) {
        float[] values = {-1000, -9, -4, -2.87f, -.75f, -.3f, .3f, .75f, 2.87f, 9, 1000};
        Tensor input = Tensor.from(values, dtype, 11);
        float[][] reference = reference(input);
        input.requiresGrad(true);
        Tensor output = ModelOps.silu(input);
        assertArrayEquals(Tensor.from(reference[0], dtype, 11).toFloatArray(), output.toFloatArray(), 1e-6f);
        output.sumTensor().backward();
        assertEquals(DType.FLOAT32, input.grad().dtype());
        assertArrayEquals(reference[1], input.grad().toFloatArray(), dtype == DType.FLOAT32 ? 1e-6f : 3e-5f);
        try (var ignored = GradMode.noGrad()) {
            assertArrayEquals(output.toFloatArray(), ModelOps.silu(input).toFloatArray());
            assertEquals(output.toFloatArray()[3], ModelOps.silu(input.select(0, 3)).item(), 1e-6f);
        }
    }

    private static float[][] reference(Tensor input) {
        float[] values = input.toFloatArray();
        float[] expected = new float[values.length], derivative = new float[values.length];
        for (int index = 0; index < values.length; index++) {
            float value = values[index], sigmoid = 1 / (1 + (float) Math.exp(-value));
            expected[index] = value * sigmoid;
            derivative[index] = sigmoid + expected[index] * (1 - sigmoid);
        }
        return new float[][]{expected, derivative};
    }
}
