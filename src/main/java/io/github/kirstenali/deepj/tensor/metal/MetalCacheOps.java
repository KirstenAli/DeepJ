package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;

abstract class MetalCacheOps extends MetalTrainingOps {
    @Override
    public void writeHeadCache(Tensor cache, Tensor update, int heads, int offset) {
        HeadCacheOps.requireWrite(cache, update, heads, offset);
        GpuBuffer destination = gpuIn(cache);
        graph.recordHeadCacheWrite(gpuIn(update), destination, heads, offset);
        destination.cpuStale = true;
    }

    @Override
    public Tensor headPrefixMatmul(Tensor left, Tensor cache, int heads, int length, boolean transposeCache) {
        HeadCacheOps.requirePrefix(cache, heads, length);
        requireDivisible(left.rows, heads, "query rows");
        int inner = transposeCache ? cache.cols : length;
        if (left.cols != inner) throw new IllegalArgumentException("Head cache matmul shape mismatch");
        int columns = transposeCache ? length : cache.cols;
        GpuBuffer output = graph.newOutputBuffer(left.rows, columns);
        graph.recordHeadPrefixMatmul(gpuIn(left), gpuIn(cache), output, heads, length, transposeCache);
        return gpuOut(output);
    }
}
