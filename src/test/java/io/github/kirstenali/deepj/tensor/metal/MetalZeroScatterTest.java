package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MetalZeroScatterTest extends MetalDifferentialTestSupport {
    @Test
    void explicitZeroingClearsEveryBitOfExistingStorage() {
        Tensor tensor = metal.multiplyScalar(Tensor.ones(3, 5), 42);
        tensor.materialize();
        var buffer = (io.github.kirstenali.deepj.tensor.GpuBuffer) tensor.getGpuTag();
        MetalNative.nativeZeroBuffer(buffer.id, 15);
        float[] values = new float[15];
        MetalNative.nativeDownloadBuffer(buffer.id, values);
        for (float value : values) {
            assertEquals(0, Float.floatToRawIntBits(value));
        }
    }

    @Test
    void nativeZeroingRejectsMissingBuffersAndInvalidSizes() {
        Tensor tensor = metal.multiplyScalar(Tensor.ones(1, 2), 3);
        tensor.materialize();
        int id = ((io.github.kirstenali.deepj.tensor.GpuBuffer) tensor.getGpuTag()).id;
        assertThrows(RuntimeException.class, () -> MetalNative.nativeZeroBuffer(-1, 2));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeZeroBuffer(id, 0));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeZeroBuffer(id, -1));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeZeroBuffer(id, 1));
        assertArrayEquals(new float[]{3, 3}, tensor.data());
    }

    @Test
    void scatterPreservesUntouchedRowsAccumulationAndRecordTimeValues() {
        Tensor target = Tensor.zeros(4, 2).retainDeviceBuffer();
        Tensor contribution = Tensor.from2D(new float[][]{{1, 2}, {3, 4}, {5, 6}});
        metal.scatterAddRows(target, new int[]{1, 1, 3}, contribution);
        contribution.rawData()[0] = 900;
        target.rawData()[0] = 800;
        metal.scatterAddRows(target, new int[]{1}, Tensor.ones(1, 2));
        metal.releaseTemporaryResources();
        assertArrayEquals(new float[]{0, 0, 5, 7, 0, 0, 5, 6}, target.data());
    }

    @Test
    void nonzeroAndNegativeZeroScatterTargetsKeepTheirValues() {
        Tensor target = Tensor.from2D(new float[][]{{-0.0f, 9}, {2, 3}});
        metal.scatterAddRows(target, new int[]{1}, Tensor.ones(1, 2));
        assertArrayEquals(new float[]{-0.0f, 9, 3, 4}, target.data());
        assertEquals(0x80000000, Float.floatToRawIntBits(target.rawData()[0]));
    }

    @Test
    void zeroScatterCanBeMaterializedAfterSwitchingBackends() {
        Tensor target = Tensor.zeros(2, 2);
        metal.scatterAddRows(target, new int[]{0}, Tensor.ones(1, 2));
        Tensor.setBackend(cpu);
        assertArrayEquals(new float[]{1, 1, 0, 0}, target.data());
        Tensor.setBackend(metal);
    }
}
