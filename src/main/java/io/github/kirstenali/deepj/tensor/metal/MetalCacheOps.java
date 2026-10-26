package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;

abstract class MetalCacheOps extends MetalTrainingOps {
    @Override
    public void writeHeadCache(Tensor cache, Tensor update, int heads, int offset) {
        HeadCacheOps.requireWrite(cache, update, heads, offset);
        GpuBuffer destination = gpuIn(cache);
        graph.recordHeadCacheWrite(gpuIn(update), destination, heads, update.matrixRows() / heads,
                cache.matrixRows() / heads, cache.matrixCols(), offset);
        graph.markUpdated(cache);
    }

    @Override
    public Tensor headPrefixMatmul(Tensor left, Tensor cache, int heads, int length, boolean transposeCache) {
        HeadCacheOps.requirePrefix(cache, heads, length);
        requireDivisible(left.matrixRows(), heads, "query rows");
        int inner = transposeCache ? cache.matrixCols() : length;
        if (left.matrixCols() != inner) throw new IllegalArgumentException("Head cache matmul shape mismatch");
        int columns = transposeCache ? length : cache.matrixCols();
        GpuBuffer output = graph.newOutputBuffer(left.matrixRows(), columns);
        graph.recordHeadPrefixMatmul(gpuIn(left), gpuIn(cache), output, heads, length, transposeCache,
                left.matrixRows(), left.matrixCols(), cache.matrixCols());
        return gpuOut(output);
    }
}
