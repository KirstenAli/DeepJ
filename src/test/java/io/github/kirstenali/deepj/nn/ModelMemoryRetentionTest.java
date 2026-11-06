package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ModelMemoryRetentionTest {
    @BeforeEach void backend() {
        Tensor.setBackend(Boolean.getBoolean("sanavideo.test.metal") ? new MetalBackend() : new CpuBackend());
    }

    @Test void closingAPinPreservesTheCallersRetention() {
        Tensor value = Tensor.from(new float[]{1, 2}, 2).retainDeviceBuffer();
        try (var pin = ModelMemory.retain(value)) { assertTrue(value.retainsDeviceBuffer()); }
        assertTrue(value.retainsDeviceBuffer());
        value.releaseDeviceBufferRetention();
        assertFalse(value.retainsDeviceBuffer());
    }

    @Test void overlappingPinsOwnIndependentRetentions() {
        Tensor value = Tensor.from(new float[]{1, 2}, 2);
        var first = ModelMemory.retain(value);
        var second = ModelMemory.retain(value);
        first.close();
        assertTrue(value.retainsDeviceBuffer());
        second.close();
        second.close();
        assertFalse(value.retainsDeviceBuffer());
    }

    @Test void nestedPinsPreserveTheOuterLeaseForStridedTensors() {
        Tensor value = Tensor.from(new float[]{1, 2, 3, 4}, 2, 2).transpose();
        try (var outer = ModelMemory.retain(value)) {
            try (var inner = ModelMemory.retain(value)) { assertTrue(value.retainsDeviceBuffer()); }
            assertTrue(value.retainsDeviceBuffer());
        }
        assertFalse(value.retainsDeviceBuffer());
    }

    @Test void finishPreservesAnExistingDeviceRetention() {
        Tensor value = Tensor.from(new float[]{1, 2}, 2).retainDeviceBuffer();
        try (var ignored = GradMode.noGrad()) { assertSame(value, ModelMemory.finish(value)); }
        assertTrue(value.retainsDeviceBuffer());
        value.releaseDeviceBufferRetention();
    }

    @Test void aNestedPinKeepsComputedValuesAliveAndItsPrivateAliasDoesNotLeak() {
        try (var ignored = GradMode.noGrad(); var scope = TensorScope.open()) {
            Tensor value = Tensor.from(new float[]{1, 2}, 2).multiplyScalar(3).addScalar(1);
            try (var outer = ModelMemory.retain(value)) {
                try (var inner = ModelMemory.retain(value)) { assertTrue(value.retainsDeviceBuffer()); }
                Tensor.backend().releaseTemporaryResources();
                assertArrayEquals(new float[]{4, 7}, value.toFloatArray());
            }
            Tensor.backend().releaseTemporaryResources();
            assertFalse(value.retainsDeviceBuffer());
            assertNull(value.getGpuTag());
        }
    }

    @Test void nullArgumentsDoNotLeavePartiallyRetainedBuffers() {
        Tensor value = Tensor.from(new float[]{1, 2}, 2);
        assertThrows(NullPointerException.class, () -> ModelMemory.retain(value, null));
        assertFalse(value.retainsDeviceBuffer());
    }
}
