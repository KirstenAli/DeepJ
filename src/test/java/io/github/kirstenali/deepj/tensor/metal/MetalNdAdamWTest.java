package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class MetalNdAdamWTest extends MetalDifferentialTestSupport {
    @Test
    void ranksZeroThroughSixRetainTheExactExistingFusedMatrixUpdateBits() {
        for (int rank = 0; rank <= 6; rank++) { checkRank(rank); }
    }

    private void checkRank(int rank) {
        int[] shape = new int[rank];
        Arrays.fill(shape, 1);
        Tensor value = Tensor.from(new float[]{0.7f}, shape), reference = Tensor.from(new float[]{0.7f}, 1, 1);
        Tensor m = metal.zeros(DType.FLOAT32, shape), v = metal.zeros(DType.FLOAT32, shape);
        Tensor rm = metal.zeros(1, 1), rv = metal.zeros(1, 1);
        for (int step = 1; step <= 10; step++) {
            Tensor gradient = Tensor.from(new float[]{(step - 3) * 0.1f}, shape);
            update(value, gradient, m, v);
            update(reference, gradient.reshape(1, 1), rm, rv);
            assertArrayEquals(reference.toFloatArray(), value.toFloatArray());
            assertArrayEquals(rm.toFloatArray(), m.toFloatArray());
            assertArrayEquals(rv.toFloatArray(), v.toFloatArray());
        }
    }

    @Test
    void independentStridedStateUpdatesMatchCpuAndKeepParentFences() {
        Tensor w = Tensor.from(new float[]{9, 1, 2, 3, 4, 8}, 2, 3).slice(1, 1, 3, 1).transpose(0, 1);
        Tensor g = Tensor.from(new float[]{0.1f, 0.4f, -0.2f, 0.3f}, 2, 2).transpose(0, 1);
        Tensor m = new Tensor(3, 3).slice(0, 1, 3, 1).slice(1, 1, 3, 1);
        Tensor v = new Tensor(2, 3).slice(1, 1, 3, 1).transpose(0, 1);
        Tensor expected = new Tensor(w), first = new Tensor(m), second = new Tensor(v);
        cpu.adamWUpdate(expected, g, first, second, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.1f, 0.001f);
        update(w, g, m, v);
        assertArrayEquals(expected.toFloatArray(), w.toFloatArray(), 3e-6f);
        assertArrayEquals(first.toFloatArray(), m.toFloatArray(), 3e-6f);
        assertArrayEquals(second.toFloatArray(), v.toFloatArray(), 3e-6f);
    }

    @Test
    void partialStateWritesPreserveAllParentFences() {
        Tensor weights = Tensor.from(new float[]{9, 1, 2, 8});
        Tensor first = Tensor.from(new float[]{7, 0, 0, 6}), second = Tensor.from(new float[]{5, 0, 0, 4});
        update(weights.slice(0, 1, 3, 1), Tensor.from(new float[]{0.2f, -0.3f}),
                first.slice(0, 1, 3, 1), second.slice(0, 1, 3, 1));
        assertArrayEquals(new float[]{9, 8, 7, 6, 5, 4}, new float[]{weights.get(0), weights.get(3),
                first.get(0), first.get(3), second.get(0), second.get(3)});
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void compactParametersAccumulateUnroundedFloat32MastersAndResumeExactState(DType dtype) {
        Parameter parameter = parameter(dtype), reference = parameter(DType.FLOAT32);
        AdamW optimizer = optimizer(), expected = optimizer();
        for (int step = 0; step < 40; step++) {
            optimizer.step(List.of(parameter));
            expected.step(List.of(reference));
        }
        var state = optimizer.state(List.of(parameter));
        assertArrayEquals(reference.value.toFloatArray(), state.masterWeights().get(0).toFloatArray());
        assertArrayEquals(reference.value.to(dtype).toFloatArray(), parameter.value.toFloatArray());
        assertExactContinuation(parameter, optimizer, state);
    }

    private static void assertExactContinuation(Parameter parameter, AdamW optimizer, AdamW.State state) {
        Parameter resumed = new Parameter(parameter.value.detach());
        resumed.setGradient(parameter.grad().detach());
        AdamW continuation = optimizer();
        continuation.restoreState(List.of(resumed), state);
        optimizer.step(List.of(parameter));
        continuation.step(List.of(resumed));
        assertArrayEquals(parameter.value.toFloatArray(), resumed.value.toFloatArray());
        assertArrayEquals(optimizer.state(List.of(parameter)).masterWeights().get(0).toFloatArray(),
                continuation.state(List.of(resumed)).masterWeights().get(0).toFloatArray());
    }

    @Test
    void compactAutogradToyTrainingReducesLossAndInvalidStateCannotAdvanceVersions() {
        Parameter parameter = parameter(DType.FLOAT16);
        AdamW optimizer = AdamW.defaultAdamW(0.01f);
        float initial = parameter.value.pow(2).sumTensor().item();
        for (int step = 0; step < 20; step++) {
            parameter.zeroGrad();
            Tensor loss = parameter.value.pow(2).sumTensor();
            loss.backward();
            optimizer.step(List.of(parameter));
        }
        assertTrue(parameter.value.pow(2).sumTensor().item() < initial);
        Tensor w = new Tensor(2, 1, 3), g = new Tensor(w.shape()), m = new Tensor(w.shape());
        assertThrows(IllegalArgumentException.class, () -> update(w, g, m, m));
        assertEquals(0, w.gradientState().version());
        assertEquals(0, m.gradientState().version());
    }

    private void update(Tensor w, Tensor g, Tensor m, Tensor v) {
        metal.adamWUpdate(w, g, m, v, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.1f, 0.001f);
    }
    private static Parameter parameter(DType dtype) {
        Parameter parameter = new Parameter(Tensor.from(new float[]{1, 2, 3, 4}, dtype, 2, 1, 2));
        parameter.setGradient(Tensor.from(new float[]{0.2f, -0.1f, 0.4f, -0.3f}, parameter.value.shape()));
        return parameter;
    }
    private static AdamW optimizer() { return new AdamW(0.0001f, 0.9f, 0.999f, 1e-8f, 0); }
}
