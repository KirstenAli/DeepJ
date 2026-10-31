package io.github.kirstenali.deepj.optimisers;

import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class NdAdamWTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void rankZeroThroughSixRetainExactMatrixAdamArithmeticWithoutMasterDuplicates() {
        for (int rank = 0; rank <= 6; rank++) {
            checkShape(rank);
        }
    }

    private static void checkShape(int rank) {
        int[] shape = new int[rank];
        java.util.Arrays.fill(shape, 1);
        Parameter parameter = new Parameter(Tensor.from(new float[]{0.7f}, shape));
        Parameter reference = new Parameter(Tensor.from(new float[]{0.7f}, 1, 1));
        AdamW actual = AdamW.defaultAdamW(0.01f), expected = AdamW.defaultAdamW(0.01f);
        for (int step = 0; step < 10; step++) {
            checkStep(parameter, reference, actual, expected, (step - 3) * 0.1f);
        }
        var state = actual.state(List.of(parameter));
        assertTrue(state.masterWeights().isEmpty());
        assertArrayEquals(shape, state.firstMoments().get(0).shape());
    }

    private static void checkStep(Parameter parameter, Parameter reference,
                                    AdamW actual, AdamW expected, float gradient) {
        parameter.setGradient(Tensor.from(new float[]{gradient}, parameter.value.shape()));
        reference.setGradient(Tensor.from(new float[]{gradient}, 1, 1));
        actual.step(List.of(parameter));
        expected.step(List.of(reference));
        assertEquals(reference.value.item(), parameter.value.item());
    }

    @Test
    void stridedParametersAndMomentsUseLogicalElementAddresses() {
        Tensor value = Tensor.from(new float[]{1, 2, 3, 4}, 2, 2).transpose(0, 1);
        Parameter parameter = new Parameter(value), reference = new Parameter(Tensor.from(value.toFloatArray(), 2, 2));
        Tensor gradient = Tensor.from(new float[]{1, 3, 2, 4}, 2, 2).transpose(0, 1);
        parameter.setGradient(gradient);
        reference.setGradient(Tensor.from(gradient.toFloatArray(), 2, 2));
        AdamW actual = AdamW.defaultAdamW(0.01f), expected = AdamW.defaultAdamW(0.01f);
        actual.step(List.of(parameter));
        expected.step(List.of(reference));
        assertArrayEquals(reference.value.toFloatArray(), parameter.value.toFloatArray());
        parameter.zeroGrad();
        assertArrayEquals(new float[4], parameter.grad().toFloatArray());
        assertArrayEquals(new float[]{1, 2, 3, 4}, gradient.toFloatArray());
    }

    @Test
    void invalidUpdateStateIsRejectedBeforeMutationVersionsAdvance() {
        Tensor weights = new Tensor(2, 3, 4), gradient = new Tensor(2, 3, 4), moment = new Tensor(2, 3, 4);
        assertThrows(IllegalArgumentException.class, () -> Tensor.adamWUpdate(weights, gradient, moment, moment,
                0.1f, 0.9f, 0.999f, 1e-8f, 0, 0.1f, 0.001f));
        assertEquals(0, weights.gradientState().version());
        assertThrows(IllegalArgumentException.class, () -> Tensor.adamWUpdate(weights,
                gradient.to(DType.FLOAT16), moment, new Tensor(weights.shape()),
                0.1f, 0.9f, 0.999f, 1e-8f, 0, 0.1f, 0.001f));
        assertEquals(0, moment.gradientState().version());
    }

    @Test
    void zeroSizeParametersAreValidAndCompactGradientBindingsAreRejected() {
        Parameter empty = new Parameter(new Tensor(0, 2, 3));
        AdamW.defaultAdamW(0.01f).step(List.of(empty));
        assertArrayEquals(new int[]{0, 2, 3}, empty.grad().shape());
        Parameter compact = new Parameter(new Tensor(DType.FLOAT16, 1, 2, 3));
        assertEquals(DType.FLOAT32, compact.grad().dtype());
        assertThrows(IllegalArgumentException.class,
                () -> compact.setGradient(new Tensor(DType.FLOAT16, compact.value.shape())));
    }
}
