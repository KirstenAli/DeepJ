package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.TensorTestStorage;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MetalNdUnaryTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(DType.class)
    void scalarActivationAndPowerKernelsMatchQuantizedCpuReferences(DType dtype) {
        Tensor source = Tensor.from(new float[]{0.25f, 0.5f, 1, 2, 4, 8}, dtype, 2, 3);
        Tensor view = source.permute(1, 0).unsqueeze(1);
        for (int operation = 0; operation < 13; operation++) { compareUnary(view, operation); }
    }

    private void compareUnary(Tensor input, int operation) {
        if (operation == 7) input = cpu.divideScalar(input, 4); // Keep exp outputs within this absolute-error test.
        Tensor expected = unary(cpu, input, operation), actual = unary(metal, input, operation);
        assertFalse(TensorTestStorage.hasHostStorage(actual));
        assertEquals(input.dtype(), actual.dtype());
        assertArrayEquals(input.shape(), actual.shape());
        float tolerance = input.dtype() == DType.FLOAT32 ? 3e-6f : 0;
        assertArrayEquals(expected.toFloatArray(), actual.toFloatArray(), tolerance);
    }

    @Test
    void largeExponentialValuesExactlyMatchTheUnchangedFloat32MatrixKernel() {
        Tensor matrix = Tensor.from(new float[]{4, 6, 8, 10}, 2, 2);
        Tensor general = matrix.unsqueeze(0);
        assertArrayEquals(metal.exp(matrix).toFloatArray(), metal.exp(general).toFloatArray());
        assertArrayEquals(metal.gelu(matrix).toFloatArray(), metal.gelu(general).toFloatArray());
    }

    private static Tensor unary(TensorBackend backend, Tensor input, int operation) {
        if (operation < 5) return scalarOrPower(backend, input, operation);
        if (operation >= 10) return activation(backend, input, operation);
        return switch (operation) {
            case 5 -> backend.sqrt(input);
            case 6 -> backend.neg(input);
            case 7 -> backend.exp(input);
            case 8 -> backend.log(input);
            default -> backend.tanh(input);
        };
    }

    private static Tensor activation(TensorBackend backend, Tensor input, int operation) {
        return switch (operation) {
            case 10 -> backend.sigmoid(input);
            case 11 -> backend.relu(input);
            default -> backend.gelu(input);
        };
    }

    private static Tensor scalarOrPower(TensorBackend backend, Tensor input, int operation) {
        return switch (operation) {
            case 0 -> backend.addScalar(input, 0.5f);
            case 1 -> backend.multiplyScalar(input, 0.5f);
            case 2 -> backend.divideScalar(input, 2);
            case 3 -> backend.pow(input, 2);
            default -> backend.clamp(input, 0.5f, 4);
        };
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void reluAndGeluAdjointsAcceptStridedSavedInputsAndFloat32Gradients(DType dtype) {
        Tensor input = Tensor.from(new float[]{-2, -0.5f, 0, 0.5f, 1, 2}, dtype, 2, 3).permute(1, 0);
        Tensor gradient = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, 3, 2);
        Tensor relu = metal.reluBackward(input, gradient), gelu = metal.geluBackward(input, gradient);
        assertEquals(DType.FLOAT32, relu.dtype());
        assertEquals(DType.FLOAT32, gelu.dtype());
        assertArrayEquals(cpu.reluBackward(input, gradient).toFloatArray(), relu.toFloatArray());
        assertArrayEquals(cpu.geluBackward(input, gradient).toFloatArray(), gelu.toFloatArray(), 3e-6f);
    }
}
