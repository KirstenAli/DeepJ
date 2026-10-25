package io.github.kirstenali.deepj.inference;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class KeyValueCacheStorageTest {
    private final TensorBackend previous = Tensor.backend();
    private MetalBackend metal;

    @AfterEach
    void restoreBackend() {
        if (metal != null) metal.releaseResources();
        Tensor.setBackend(previous);
    }

    @Test
    void cpuCacheStorageRemainsEagerAndZeroInitialized() throws Exception {
        Tensor.setBackend(new CpuBackend());
        try (var cache = new KeyValueCache(8, 1, 2)) {
            assertArrayEquals(new float[16], tensor(cache, "keys").rawData());
            assertArrayEquals(new float[16], tensor(cache, "values").rawData());
        }
    }

    @Test
    void privateMetalCachesHaveNoHostArraysAndUnwrittenSlotsRemainZero() throws Exception {
        useMetal();
        try (var cache = new KeyValueCache(8, 1, 2)) {
            Tensor keys = tensor(cache, "keys"), values = tensor(cache, "values");
            assertNull(keys.rawData());
            assertNull(values.rawData());
            Tensor row = Tensor.from2D(new float[][]{{3, 5}});
            cache.attend(row, row, row, 1, 0, 1).materialize();
            float[] expected = new float[16];
            System.arraycopy(row.rawData(), 0, expected, 0, 2);
            assertArrayEquals(expected, keys.data());
            assertArrayEquals(expected, values.data());
            cache.attend(row, row, row, 1, 1, 1).materialize();
            assertNull(keys.rawData());
            assertNull(values.rawData());
        }
    }

    @Test
    void closingTheRequestReleasesLazyCacheRetention() throws Exception {
        useMetal();
        var cache = new KeyValueCache(8, 1, 2);
        Tensor keys = tensor(cache, "keys"), values = tensor(cache, "values");
        assertTrue(keys.retainsDeviceBuffer());
        cache.close();
        assertFalse(keys.retainsDeviceBuffer());
        assertFalse(values.retainsDeviceBuffer());
        metal.releaseTemporaryResources();
        assertNull(keys.getGpuTag());
        assertNull(values.getGpuTag());
    }

    private void useMetal() {
        assumeTrue(MetalBackend.isAvailable(), "Metal device not available");
        metal = new MetalBackend();
        Tensor.setBackend(metal);
    }

    private static Tensor tensor(KeyValueCache cache, String name) throws Exception {
        var field = KeyValueCache.class.getDeclaredField(name);
        field.setAccessible(true);
        return (Tensor) field.get(cache);
    }
}
