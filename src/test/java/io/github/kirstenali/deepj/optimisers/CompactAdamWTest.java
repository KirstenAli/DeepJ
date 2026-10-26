package io.github.kirstenali.deepj.optimisers;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class CompactAdamWTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void smallUpdatesAccumulateInFloat32MastersRatherThanDisappearingAtEachCast() {
        for (DType type : new DType[]{DType.FLOAT16, DType.BFLOAT16}) {
            checkSmallUpdates(type);
        }
    }

    private static void checkSmallUpdates(DType type) {
        Parameter parameter = parameter(type), reference = parameter(DType.FLOAT32);
        AdamW optimizer = optimizer(), expected = optimizer();
        for (int step = 0; step < 100; step++) {
            optimizer.step(List.of(parameter));
            expected.step(List.of(reference));
            Tensor master = optimizer.state(List.of(parameter)).masterWeights().get(0);
            assertArrayEquals(reference.value.toFloatArray(), master.toFloatArray());
            assertArrayEquals(reference.value.to(type).toFloatArray(), parameter.value.toFloatArray());
            assertEquals(DType.FLOAT32, optimizer.state(List.of(parameter)).firstMoments().get(0).dtype());
        }
        assertNotEquals(1, parameter.value.get(0, 0, 0));
        assertEquals(8, parameter.value.storageBytes());
        parameter.zeroGrad();
        assertArrayEquals(new float[4], parameter.grad().toFloatArray());
    }

    @Test
    void restoringStatePreservesUnroundedMastersAndClonesAllOptimizerBuffers() {
        Parameter original = parameter(DType.BFLOAT16);
        AdamW first = optimizer();
        repeat(first, original, 17);
        var state = first.state(List.of(original));
        Parameter resumed = new Parameter(original.value.detach());
        resumed.setGradient(original.grad().detach());
        AdamW second = optimizer();
        second.restoreState(List.of(resumed), state);
        assertNotSame(state.masterWeights().get(0), second.state(List.of(resumed)).masterWeights().get(0));
        repeat(first, original, 29);
        repeat(second, resumed, 29);
        assertArrayEquals(original.value.toFloatArray(), resumed.value.toFloatArray());
        assertArrayEquals(first.state(List.of(original)).masterWeights().get(0).toFloatArray(),
                second.state(List.of(resumed)).masterWeights().get(0).toFloatArray());
        assertEquals(first.stepCount(), second.stepCount());
    }

    @Test
    void compactStateRequiresCompleteFloat32MastersAndMoments() {
        Parameter parameter = parameter(DType.FLOAT16);
        AdamW optimizer = optimizer();
        var state = optimizer.state(List.of(parameter));
        var missing = new AdamW.State(0, 0.0001f, 0.9f, 0.999f, 1e-8f, 0,
                state.firstMoments(), state.secondMoments());
        assertThrows(IllegalArgumentException.class, () -> optimizer.restoreState(List.of(parameter), missing));
        var wrongType = new AdamW.State(0, 0.0001f, 0.9f, 0.999f, 1e-8f, 0,
                state.firstMoments(), state.secondMoments(), Map.of(0, parameter.value));
        assertThrows(IllegalArgumentException.class, () -> optimizer.restoreState(List.of(parameter), wrongType));
        Parameter ordinary = parameter(DType.FLOAT32);
        assertThrows(IllegalArgumentException.class, () -> optimizer.restoreState(List.of(ordinary), state));
    }

    @Test
    void explicitWeightChangesCannotBeSilentlyOverwrittenByAStaleMaster() {
        Parameter parameter = parameter(DType.FLOAT16);
        AdamW optimizer = optimizer();
        optimizer.step(List.of(parameter));
        try (var ignored = GradMode.noGrad()) { parameter.value.setAt(2, 0, 0, 0); }
        assertThrows(IllegalStateException.class, () -> optimizer.step(List.of(parameter)));
        assertEquals(2, parameter.value.get(0, 0, 0));
    }

    @Test
    void compactAutogradTrainingUsesFloat32GradientsAndReducesTheToyObjective() {
        for (DType dtype : new DType[]{DType.FLOAT16, DType.BFLOAT16}) {
            checkTraining(dtype);
        }
    }

    private static void checkTraining(DType dtype) {
        Parameter parameter = parameter(dtype);
        AdamW optimizer = AdamW.defaultAdamW(0.01f);
        float initial = parameter.value.multiply(parameter.value).sumTensor().item();
        for (int step = 0; step < 40; step++) {
            parameter.zeroGrad();
            Tensor loss = parameter.value.multiply(parameter.value).sumTensor();
            assertTrue(Float.isFinite(loss.item()));
            loss.backward();
            assertEquals(DType.FLOAT32, parameter.grad().dtype());
            optimizer.step(List.of(parameter));
        }
        assertTrue(parameter.value.multiply(parameter.value).sumTensor().item() < initial);
    }

    private static Parameter parameter(DType dtype) {
        Parameter result = new Parameter(Tensor.from(new float[]{1, 2, 3, 4}, dtype, 2, 1, 2));
        result.setGradient(Tensor.from(new float[]{0.1f, -0.2f, 0.3f, 0.4f}, result.value.shape()));
        return result;
    }

    private static AdamW optimizer() { return new AdamW(0.0001f, 0.9f, 0.999f, 1e-8f, 0); }
    private static void repeat(AdamW optimizer, Parameter parameter, int steps) {
        for (int step = 0; step < steps; step++) {
            optimizer.step(List.of(parameter));
        }
    }
}
