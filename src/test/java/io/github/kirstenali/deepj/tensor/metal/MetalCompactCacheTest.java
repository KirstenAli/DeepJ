package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.HeadCacheOps;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class MetalCompactCacheTest extends MetalDifferentialTestSupport {
    @ParameterizedTest @EnumSource(DType.class)
    void capacityStridesGroupedQueriesAndUnusedPoisonedSlotsMatchGatheredProducts(DType dtype) {
        for (int length : new int[]{1, 7, 65, 129}) {
            for (int groups : new int[]{1, 2, 5}) { check(dtype, groups, length); }
        }
    }

    private void check(DType dtype, int groups, int length) {
        int capacity = length + 17, width = 32;
        float[] values = new float[2 * capacity * width];
        var random = new Random(182 + length);
        for (int i = 0; i < values.length; i++) {
            values[i] = i / width % capacity >= length ? Float.NaN : random.nextFloat() - .5f;
        }
        Tensor cache = Tensor.from(values, dtype, 2 * capacity, width);
        Tensor query = Tensor.random(dtype, random, 2 * groups, width);
        compare(query, cache, length, true);
        compare(Tensor.random(dtype, random, 2 * groups, length), cache, length, false);
        assertArrayEquals(valuesRounded(values, dtype), cache.toFloatArray());
        Tensor padded = Tensor.from(new float[(2 * capacity + 2) * (width + 2)], dtype,
                2 * capacity + 2, width + 2).slice(0, 1, 2 * capacity + 1).slice(1, 1, width + 1);
        metal.copyInto(padded, cache);
        compare(query, padded, length, true);
    }

    private void compare(Tensor query, Tensor cache, int length, boolean transpose) {
        Tensor gathered = HeadCacheOps.prefix(metal, cache, 2, length);
        Tensor expected = metal.batchedMatmul(query, gathered, 2, false, transpose);
        Tensor actual = metal.headPrefixMatmul(query, cache, 2, length, transpose);
        assertEquals(query.dtype(), actual.dtype());
        assertTensorClose(expected, actual, .00002f, .002f);
        for (float value : actual.toFloatArray()) { assertTrue(Float.isFinite(value)); }
    }

    private static float[] valuesRounded(float[] values, DType dtype) {
        return Tensor.from(Arrays.copyOf(values, values.length), dtype, 2, values.length / 2).toFloatArray();
    }
}
