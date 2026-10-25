package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class HeadCacheOpsTest {
    @BeforeEach
    void useCpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void appendsAreHeadMajorAndPrefixProductsIgnoreUnusedCapacity() {
        Tensor cache = new Tensor(8, 1);
        Tensor.backend().writeHeadCache(cache, column(1, 2, 10, 20), 2, 0);
        Tensor.backend().writeHeadCache(cache, column(3, 30), 2, 2);
        assertArrayEquals(new float[]{1, 2, 3, 0, 10, 20, 30, 0}, cache.data);
        Tensor query = column(2, 3);
        assertArrayEquals(new float[]{2, 4, 6, 30, 60, 90},
                Tensor.backend().headPrefixMatmul(query, cache, 2, 3, true).data);
        Tensor probabilities = Tensor.from2D(new float[][]{{0, 0, 1}, {1, 0, 0}});
        assertArrayEquals(new float[]{3, 10},
                Tensor.backend().headPrefixMatmul(probabilities, cache, 2, 3, false).data);
    }

    @Test
    void invalidWritesPrefixesAndTrackedDestinationsAreRejected() {
        Tensor cache = new Tensor(8, 2), update = new Tensor(2, 2);
        assertThrows(IllegalArgumentException.class, () -> HeadCacheOps.requireWrite(cache, update, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> HeadCacheOps.requireWrite(cache, update, 3, 0));
        assertThrows(IllegalArgumentException.class, () -> HeadCacheOps.requireWrite(cache, update, 2, -1));
        assertThrows(IllegalArgumentException.class, () -> HeadCacheOps.requireWrite(cache, update, 2, 4));
        assertThrows(IllegalArgumentException.class,
                () -> HeadCacheOps.requireWrite(cache, new Tensor(3, 2), 2, 0));
        assertThrows(IllegalArgumentException.class,
                () -> HeadCacheOps.requireWrite(cache, new Tensor(2, 1), 2, 0));
        assertThrows(IllegalArgumentException.class, () -> HeadCacheOps.requirePrefix(cache, 2, 0));
        assertThrows(IllegalArgumentException.class, () -> HeadCacheOps.requirePrefix(cache, 2, 5));
        cache.requiresGrad(true);
        assertThrows(IllegalArgumentException.class, () -> HeadCacheOps.requireWrite(cache, update, 2, 0));
    }

    private static Tensor column(float... values) {
        Tensor tensor = new Tensor(values.length, 1);
        System.arraycopy(values, 0, tensor.data, 0, values.length);
        return tensor;
    }
}
