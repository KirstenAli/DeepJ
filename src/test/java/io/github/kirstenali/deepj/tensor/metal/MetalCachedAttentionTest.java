package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class MetalCachedAttentionTest extends MetalDifferentialTestSupport {
    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void shortGroupedCacheAttentionMatchesTheSameRoundedUnfusedOperations(DType dtype) {
        try (var ignored = GradMode.noGrad()) {
            for (int length : new int[]{1, 65, 129, 512}) {
                for (int width : new int[]{32, 128}) { compare(dtype, length, width); }
            }
        }
    }

    private void compare(DType type, int length, int width) {
        Tensor q = values(type, 4, width), k = cache(type, length, width, 17);
        Tensor v = cache(type, length, width, 29);
        float[] original = k.toFloatArray();
        assertTrue(metal.supportsCachedAttention(q, k, v, 2, length, .125f));
        Tensor expected = reference(q, k, v, length);
        Tensor actual = metal.cachedAttention(q, k, v, 2, length, .125f);
        assertEquals(type, actual.dtype());
        assertTensorClose(expected, actual, .0002f, type == DType.BFLOAT16 ? .02f : .003f);
        assertArrayEquals(original, k.toFloatArray());
    }

    @Test void unsupportedTypesShapesAndTrainingRetainTheirExistingRoutes() {
        Tensor q = values(DType.FLOAT16, 4, 128), k = values(DType.FLOAT16, 2 * 600, 128);
        assertFalse(metal.supportsCachedAttention(q, k, k, 2, 80, .125f));
        try (var ignored = GradMode.noGrad()) {
            assertFalse(metal.supportsCachedAttention(q.to(DType.FLOAT32), k, k, 2, 80, .125f));
            assertFalse(metal.supportsCachedAttention(q, k.to(DType.BFLOAT16), k, 2, 80, .125f));
            assertFalse(metal.supportsCachedAttention(q, k, k.to(DType.BFLOAT16), 2, 80, .125f));
            assertFalse(metal.supportsCachedAttention(q, k, k, 0, 80, .125f));
            assertFalse(metal.supportsCachedAttention(q, k, k, 3, 80, .125f));
            assertFalse(metal.supportsCachedAttention(q, k, k, 2, 513, .125f));
            assertFalse(metal.supportsCachedAttention(q, k, k, 2, 0, .125f));
            assertFalse(metal.supportsCachedAttention(q, k, k, 2, 80, Float.NaN));
            assertFalse(metal.supportsCachedAttention(q.slice(0, 1, 3), k, k, 2, 80, .125f));
            assertThrows(IllegalArgumentException.class, () -> metal.cachedAttention(q, k, k, 2, 700, .125f));
        }
    }

    @Test void nonfiniteScoresPreserveUnfusedNanResults() {
        try (var ignored = GradMode.noGrad()) {
            Tensor q = Tensor.from(new float[]{Float.NaN, 1}, DType.FLOAT16, 2, 1);
            Tensor k = Tensor.ones(DType.FLOAT16, 8, 1), v = Tensor.ones(DType.FLOAT16, 8, 1);
            assertArrayEquals(reference(q, k, v, 3).toFloatArray(),
                    metal.cachedAttention(q, k, v, 2, 3, .125f).toFloatArray());
        }
    }

    @Test void leadingViewsUseTheirLogicalShapeWithoutReadingUnusedStorage() {
        try (var ignored = GradMode.noGrad()) {
            Tensor q = values(DType.FLOAT16, 8, 32).slice(0, 0, 4);
            Tensor k = values(DType.FLOAT16, 200, 32).slice(0, 0, 140);
            Tensor v = values(DType.FLOAT16, 200, 32).slice(0, 0, 140);
            assertTrue(metal.supportsCachedAttention(q, k, v, 2, 65, .125f));
            assertTensorClose(reference(q, k, v, 65), metal.cachedAttention(q, k, v, 2, 65, .125f),
                    .0002f, .003f);
        }
    }

    private Tensor reference(Tensor q, Tensor k, Tensor v, int length) {
        Tensor scores = metal.headPrefixMatmul(q, k, 2, length, true);
        Tensor probabilities = metal.softmaxRows(scores.multiplyScalar(.125f));
        return metal.headPrefixMatmul(probabilities, v, 2, length, false);
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void compactReadsPreserveEveryRepresentableValueIncludingSubnormals(DType dtype) {
        float[] values = new float[65536];
        for (int i = 0; i < values.length; i++) {
            values[i] = dtype == DType.FLOAT16 ? Float.float16ToFloat((short) i) : Float.intBitsToFloat(i << 16);
        }
        try (var ignored = GradMode.noGrad()) {
            Tensor q = Tensor.zeros(dtype, 512, 128), v = Tensor.from(values, dtype, 512, 128);
            Tensor scores = metal.headPrefixMatmul(q, q, 512, 1, true);
            Tensor expected = metal.headPrefixMatmul(metal.softmaxRows(scores), v, 512, 1, false);
            assertArrayEquals(expected.toFloatArray(),
                    metal.cachedAttention(q, q, v, 512, 1, .125f).toFloatArray(), 0);
        }
    }

    private static Tensor cache(DType type, int length, int width, long seed) {
        int capacity = length + 5;
        float[] values = new float[2 * capacity * width];
        var random = new Random(seed);
        for (int i = 0; i < values.length; i++) {
            values[i] = i / width % capacity >= length ? Float.NaN : random.nextFloat() - .5f;
        }
        return Tensor.from(values, type, 2 * capacity, width);
    }

    private static Tensor values(DType type, int rows, int columns) {
        return Tensor.random(type, new Random(671 + rows + columns), rows, columns).multiplyScalar(.25f);
    }
}
