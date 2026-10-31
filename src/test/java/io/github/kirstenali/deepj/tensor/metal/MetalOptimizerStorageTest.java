package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class MetalOptimizerStorageTest extends MetalDifferentialTestSupport {
    @Test
    void backendZerosAreLazyWhilePublicTensorZerosRemainCpuBacked() {
        Tensor state = metal.zeros(3, 5).retainDeviceBuffer();
        assertNull(state.rawData());
        assertNotNull(Tensor.zeros(3, 5).rawData());
        metal.releaseTemporaryResources();
        assertArrayEquals(new float[15], state.data());
    }

    @Test
    void materializedMomentShadowsAreReleasedAgainOnTheNextUpdate() {
        Parameter parameter = parameter();
        AdamW optimizer = AdamW.defaultAdamW(1e-3f);
        optimizer.step(List.of(parameter));
        var state = optimizer.state(List.of(parameter));
        Tensor first = state.firstMoments().get(0), second = state.secondMoments().get(0);
        assertLazyMoments(state);
        assertArrayEquals(new float[]{0.1f, -0.1f}, first.data(), 1e-7f);
        second.materialize();
        optimizer.step(List.of(parameter));
        metal.releaseTemporaryResources();
        assertLazyMoments(state);
        assertNotNull(parameter.value.rawData());
        assertTrue(first.retainsDeviceBuffer());
        assertArrayEquals(new float[]{0.19f, -0.19f}, first.data(), 1e-7f);
    }

    private static void assertLazyMoments(AdamW.State state) {
        assertNull(state.firstMoments().get(0).rawData());
        assertNull(state.secondMoments().get(0).rawData());
    }

    @Test
    void restoringMomentsCopiesAllBitsAndDoesNotAliasTheSuppliedState() {
        Parameter parameter = parameter();
        Tensor first = Tensor.from2D(new float[][]{{-0.0f, Float.MIN_VALUE}});
        Tensor second = Tensor.zeros(1, 2);
        var supplied = new AdamW.State(3, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f,
                List.of(first), List.of(second));
        AdamW optimizer = AdamW.defaultAdamW(1e-3f);
        optimizer.restoreState(List.of(parameter), supplied);
        Tensor restored = optimizer.state(List.of(parameter)).firstMoments().get(0);
        first.rawData()[1] = 9;
        assertNull(restored.rawData());
        assertArrayEquals(new float[]{-0.0f, Float.MIN_VALUE}, restored.data());
        assertEquals(0x80000000, Float.floatToRawIntBits(restored.rawData()[0]));
        assertNotSame(first, restored);
    }

    @Test
    void deviceCopiesHandleGpuInputsAndBackendSwitchesWithoutAliasing() {
        Tensor source = metal.multiplyScalar(Tensor.ones(1, 2), 3);
        Tensor copy = metal.copy(source).retainDeviceBuffer();
        source.addScalarInPlace(7);
        Tensor.setBackend(cpu);
        assertArrayEquals(new float[]{3, 3}, copy.data());
        assertArrayEquals(new float[]{10, 10}, source.data());
        Tensor.setBackend(metal);
    }

    private Parameter parameter() {
        Parameter parameter = new Parameter(Tensor.ones(1, 2));
        parameter.setGradient(Tensor.from2D(new float[][]{{1, -1}}));
        return parameter;
    }
}
