package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertTensorGradients;
import static org.junit.jupiter.api.Assertions.*;

class NdSwiGluTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void ndSwiGluMatchesBothInputFiniteDifferences() {
        Tensor gate = gate(), up = up(), upstream = Tensor.from(new float[]{0.2f, 0.3f, 0.4f, 0.5f}, 2, 1, 2);
        assertTensorGradients(gate, value -> ActivationOps.swiGlu(value, up), upstream,
                1e-3f, 1e-4f, 1e-3f);
        assertTensorGradients(up, value -> ActivationOps.swiGlu(gate, value), upstream,
                1e-3f, 1e-4f, 1e-3f);
    }

    @Test
    void compactOutputsRoundOnceAndGradientsMatchDecodedFloat32() {
        for (DType type : DType.values()) {
            Tensor gate = gate().to(type).requiresGrad(true), up = up().to(type).requiresGrad(true);
            Tensor decoded = Tensor.from(gate.toFloatArray(), 2, 2).requiresGrad(true);
            Tensor decodedUp = Tensor.from(up.toFloatArray(), 2, 2).requiresGrad(true);
            Tensor expected = ActivationOps.swiGlu(decoded, decodedUp), output = ActivationOps.swiGlu(gate, up);
            assertArrayEquals(expected.to(type).toFloatArray(), output.toFloatArray());
            output.sumTensor().backward();
            expected.sumTensor().backward();
            assertArrayEquals(decoded.grad().toFloatArray(), gate.grad().toFloatArray());
            assertArrayEquals(decodedUp.grad().toFloatArray(), up.grad().toFloatArray());
            assertEquals(DType.FLOAT32, gate.grad().dtype());
        }
    }

    @Test
    void stridedMixedDtypesPromoteAndRetainLogicalShape() {
        Tensor gate = gate().transpose(0, 2).to(DType.FLOAT16).transpose(0, 2);
        Tensor up = up().to(DType.BFLOAT16);
        Tensor output = ActivationOps.swiGlu(gate, up);
        assertEquals(DType.FLOAT32, output.dtype());
        assertArrayEquals(gate.shape(), output.shape());
        Tensor decoded = Tensor.from(gate.toFloatArray(), 2, 2), decodedUp = Tensor.from(up.toFloatArray(), 2, 2);
        assertArrayEquals(ActivationOps.swiGlu(decoded, decodedUp).toFloatArray(), output.toFloatArray());
    }

    @Test
    void emptyAndScalarOperandsAreSupportedButBroadcastingIsNotImplicit() {
        Tensor empty = new Tensor(0, 2, 3).requiresGrad(true);
        ActivationOps.swiGlu(empty, new Tensor(0, 2, 3)).sumTensor().backward();
        assertArrayEquals(empty.shape(), empty.grad().shape());
        Tensor scalar = Tensor.from(new float[]{0.7f}, new int[0]).requiresGrad(true);
        assertEquals(0, ActivationOps.swiGlu(scalar, scalar).rank());
        assertThrows(IllegalArgumentException.class, () -> ActivationOps.swiGlu(new Tensor(1, 2), new Tensor(2)));
        assertThrows(IllegalArgumentException.class, () -> Tensor.backend().swiGluBackward(
                new Tensor(1), gate(), up()));
    }

    private static Tensor gate() { return Tensor.from(new float[]{-0.7f, 1.1f, 0.3f, 1.2f}, 2, 1, 2); }
    private static Tensor up() { return Tensor.from(new float[]{0.7f, -0.4f, 1.5f, 0.9f}, 2, 1, 2); }
}
