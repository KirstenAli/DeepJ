package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.autograd.LayoutOps;
import io.github.kirstenali.deepj.tensor.*;
import static org.junit.jupiter.api.Assertions.*;

/** Checked legacy entry points must use the same typed layouts as their ND equivalents. */
public final class PackedLayoutAssertions {
    private PackedLayoutAssertions() {}

    public static void heads(TensorBackend backend, DType dtype) {
        Tensor source = Tensor.from(new float[]{1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12}, dtype, 4, 3);
        Tensor input = backend.permute(source, 1, 0).requiresGrad(true);
        Tensor split = LayoutOps.splitHeads(input, 2), restored = LayoutOps.mergeHeads(split, 2);
        assertEquals(dtype, split.dtype());
        assertArrayEquals(new int[]{6, 2}, split.shape());
        assertArrayEquals(new float[]{1, 2, 5, 6, 9, 10, 3, 4, 7, 8, 11, 12}, split.toFloatArray());
        assertArrayEquals(input.toFloatArray(), restored.toFloatArray());
        restored.sumTensor().backward();
        for (float value : input.grad().toFloatArray()) { assertEquals(1, value); }
        assertEquals(DType.FLOAT32, input.grad().dtype());
    }

    public static void batchedProducts(TensorBackend backend, DType dtype, boolean ta, boolean tb) {
        Tensor left = Tensor.from(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, dtype, 4, 2);
        Tensor right = Tensor.from(new float[]{1, 0, 0, 1, 2, 1, 1, 2}, dtype, 4, 2);
        Tensor a = packed(backend, left, ta).requiresGrad(true), b = packed(backend, right, tb).requiresGrad(true);
        Tensor actual = LayoutOps.batchedMatmul(a, b, 2, ta, tb);
        actual.sumTensor().backward();
        assertEquals(dtype, actual.dtype());
        assertArrayEquals(new float[]{1, 2, 3, 4, 16, 17, 22, 23}, actual.toFloatArray());
        assertArrayEquals(packed(backend, Tensor.from(new float[]{1, 1, 1, 1, 3, 3, 3, 3}, 4, 2), ta)
                .toFloatArray(), a.grad().toFloatArray());
        assertArrayEquals(packed(backend, Tensor.from(new float[]{4, 4, 6, 6, 12, 12, 14, 14}, 4, 2), tb)
                .toFloatArray(), b.grad().toFloatArray());
        assertEquals(DType.FLOAT32, a.grad().dtype());
    }

    private static Tensor packed(TensorBackend backend, Tensor input, boolean transpose) {
        if (!transpose) return input;
        return backend.reshape(backend.permute(backend.reshape(input, 2, 2, 2), 0, 2, 1), 4, 2);
    }

    public static Tensor cache(TensorBackend backend, DType dtype) {
        Tensor cache = backend.reshape(backend.zeros(dtype, 2, 4, 2), 8, 2);
        Tensor initial = backend.permute(Tensor.from(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, dtype, 2, 4), 1, 0);
        backend.writeHeadCache(cache, initial, 2, 0);
        backend.writeHeadCache(cache, Tensor.from(new float[]{9, 10, 11, 12}, dtype, 2, 2), 2, 2);
        Tensor query = Tensor.from(new float[]{1, 2, 2, 1}, dtype, 2, 2);
        Tensor scores = backend.headPrefixMatmul(query, cache, 2, 3, true);
        assertArrayEquals(new float[]{11, 14, 29, 13, 16, 34}, scores.toFloatArray());
        Tensor probabilities = Tensor.from(new float[]{0, 0, 1, 1, 0, 0}, dtype, 2, 3);
        Tensor result = backend.headPrefixMatmul(probabilities, cache, 2, 3, false);
        assertArrayEquals(new float[]{9, 10, 3, 7}, result.toFloatArray());
        assertEquals(dtype, scores.dtype());
        assertEquals(dtype.storageBytes(16), cache.storageBytes());
        return cache;
    }

    public static void cacheAliasing(TensorBackend backend, DType dtype) {
        Tensor cache = Tensor.from(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, dtype, 8, 2);
        Tensor source = backend.slice(cache, 0, 0, 4, 1);
        long version = cache.gradientState().version();
        backend.writeHeadCache(cache, source, 2, 2);
        assertArrayEquals(new float[]{1, 2, 3, 4, 1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8}, cache.toFloatArray());
        assertTrue(cache.gradientState().version() > version);
        assertThrows(IllegalArgumentException.class, () -> backend.writeHeadCache(cache, source, 2, 3));
    }
}
