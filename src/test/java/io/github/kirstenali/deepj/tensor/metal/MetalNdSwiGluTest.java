package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.ActivationOps;
import io.github.kirstenali.deepj.tensor.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Arrays;

import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertTensorGradients;
import static org.junit.jupiter.api.Assertions.*;

class MetalNdSwiGluTest extends MetalDifferentialTestSupport {
    @Test
    void everyRankAndDtypeMatchesCpuWithoutRoundingCompactIntermediates() {
        for (DType dtype : DType.values()) {
            for (int rank = 0; rank <= 6; rank++) {
                int[] shape = new int[rank];
                Arrays.fill(shape, 1);
                Tensor gate = Tensor.from(new float[]{0.7f}, dtype, shape);
                Tensor up = Tensor.from(new float[]{-0.4f}, dtype, shape);
                assertArrayEquals(cpu.swiGlu(gate, up).toFloatArray(), metal.swiGlu(gate, up).toFloatArray(), 3e-6f);
            }
        }
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void stridedInputsAndFloat32AdjointsMatchDecodedReferences(DType dtype) {
        Tensor gate = Tensor.from(new float[]{0.2f, -0.4f, 0.6f, 0.8f, -0.2f, 0.1f}, dtype, 2, 3).transpose(0, 1);
        Tensor up = Tensor.from(new float[]{0.3f, 0.4f, 0.5f, 0.2f, 0.1f, -0.2f}, dtype, 3, 2);
        Tensor gradient = Tensor.from(new float[]{0.5f, 0.1f, 0.2f, 0.7f, -0.2f, 0.3f}, 3, 2);
        assertArrayEquals(cpu.swiGlu(gate, up).toFloatArray(), metal.swiGlu(gate, up).toFloatArray(), 3e-6f);
        var expected = cpu.swiGluBackward(gradient, gate, up);
        var actual = metal.swiGluBackward(gradient, gate, up);
        assertArrayEquals(expected.gateGradient().toFloatArray(), actual.gateGradient().toFloatArray(), 3e-6f);
        assertArrayEquals(expected.upGradient().toFloatArray(), actual.upGradient().toFloatArray(), 3e-6f);
        assertEquals(DType.FLOAT32, actual.gateGradient().dtype());
    }

    @Test
    void gateAndUpGradientsMatchActualGpuFiniteDifferences() {
        Tensor gate = Tensor.from(new float[]{0.1f, -0.3f, 0.4f, 0.7f}, 2, 1, 2);
        Tensor up = Tensor.from(new float[]{0.2f, 0.5f, -0.4f, 0.1f}, gate.shape());
        Tensor gradient = Tensor.from(new float[]{0.4f, -0.2f, 0.3f, 0.1f}, gate.shape());
        assertTensorGradients(gate, value -> ActivationOps.swiGlu(value, up), gradient, 1e-3f, 1e-4f, 1e-3f);
        assertTensorGradients(up, value -> ActivationOps.swiGlu(gate, value), gradient, 1e-3f, 1e-4f, 1e-3f);
    }

    @Test
    void mixedCompactOperandsPromoteOnlyTheCompletedOutput() {
        Tensor gate = Tensor.from(new float[]{0.2f, 0.7f, -0.3f}, DType.FLOAT16);
        Tensor up = Tensor.from(new float[]{-0.4f, 0.3f, 0.6f}, DType.BFLOAT16);
        Tensor output = metal.swiGlu(gate, up);
        assertEquals(DType.FLOAT32, output.dtype());
        assertArrayEquals(cpu.swiGlu(gate, up).toFloatArray(), output.toFloatArray(), 3e-6f);
    }

    @Test
    void denseFloat32NdPathsKeepTheExistingFusedKernelBitsAndHostLazyOutputs() {
        Tensor gate = metal.copy(Tensor.from(new float[]{0.1f, 0.5f, -0.2f, 0.3f}, 2, 1, 2));
        Tensor up = metal.copy(Tensor.from(new float[]{0.2f, -0.3f, 0.6f, 0.1f}, gate.shape()));
        Tensor result = metal.swiGlu(gate, up);
        metal.graph.flush();
        assertFalse(TensorTestStorage.hasHostStorage(result));
        assertArrayEquals(metal.swiGlu(gate.reshape(2, 2), up.reshape(2, 2)).toFloatArray(), result.toFloatArray());
        assertEquals(0, metal.swiGlu(new Tensor(DType.FLOAT16, 0, 3), new Tensor(DType.FLOAT16, 0, 3)).numel());
        assertThrows(IllegalArgumentException.class, () -> metal.swiGlu(gate, new Tensor(2)));
    }
}
