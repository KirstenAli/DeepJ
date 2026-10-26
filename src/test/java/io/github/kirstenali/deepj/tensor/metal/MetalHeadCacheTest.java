package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.*;

class MetalHeadCacheTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @CsvSource({"1,1,1,8", "3,3,7,16", "3,1,129,256", "9,1,31,32",
            "2,1,16,32", "2,1,256,288", "2,1,256,1024", "3,3,256,1024"})
    void deviceOnlyAppendsAndStridedProductsMatchCompactCpuReference(
            int heads, int queries, int length, int capacity) {
        Tensor cache = new Tensor(heads * capacity, 64).retainDeviceBuffer();
        Tensor initial = random(heads * length, 64, 457), next = random(heads, 64, 461);
        metal.writeHeadCache(cache, initial, heads, 0);
        Tensor.backend().releaseTemporaryResources();
        metal.writeHeadCache(cache, next, heads, length);
        Tensor query = random(heads * queries, 64, 463);
        Tensor compact = compact(initial, next, heads, length);
        Tensor scores = metal.headPrefixMatmul(query, cache, heads, length + 1, true);
        Tensor expected = cpu.batchedMatmul(query, compact, heads, false, true);
        assertTensorClose(expected, scores, 2e-5f, 2e-4f);
        assertTensorClose(cpu.batchedMatmul(expected, compact, heads, false, false),
                metal.headPrefixMatmul(scores, cache, heads, length + 1, false), 3e-5f, 5e-4f);
        cache.releaseDeviceBufferRetention();
    }

    @Test
    void shapeValidationRunsBeforeEncodingAnyCacheCommand() {
        Tensor cache = new Tensor(8, 4), query = new Tensor(2, 4);
        assertThrows(IllegalArgumentException.class, () -> metal.writeHeadCache(cache, query, 2, 4));
        assertThrows(IllegalArgumentException.class, () -> metal.headPrefixMatmul(query, cache, 2, 5, true));
        assertThrows(IllegalArgumentException.class, () -> metal.headPrefixMatmul(query, cache, 3, 1, true));
        assertThrows(IllegalArgumentException.class, () -> metal.headPrefixMatmul(query, cache, 2, 2, false));
    }

    private static Tensor compact(Tensor initial, Tensor next, int heads, int length) {
        Tensor result = new Tensor(heads * (length + 1), initial.matrixCols());
        for (int head = 0; head < heads; head++) {
            int start = head * (length + 1) * initial.matrixCols();
            System.arraycopy(initial.rawData(), head * length * initial.matrixCols(),
                    result.rawData(), start, length * initial.matrixCols());
            System.arraycopy(next.rawData(), head * initial.matrixCols(),
                    result.rawData(), start + length * initial.matrixCols(), initial.matrixCols());
        }
        return result;
    }
}
