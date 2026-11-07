package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;

abstract class MetalCacheOps extends MetalTrainingOps implements CachedAttentionBackend {
    @Override public boolean supportsCachedAttention(Tensor q, Tensor k, Tensor v, int heads, int length, float scale) {
        return MetalCachedAttention.supports(q, k, v, heads, length, scale);
    }

    @Override public Tensor cachedAttention(Tensor q, Tensor k, Tensor v, int heads, int length, float scale) {
        return MetalCachedAttention.apply(graph, q, k, v, heads, length, scale);
    }

    @Override public boolean supportsCachedAttention(Tensor q, Tensor k, Tensor v, int heads, int length,
                                                     CachedAttentionBackend.Options options) {
        return MetalCachedAttention.supports(q, k, v, heads, length, options);
    }

    @Override public Tensor cachedAttention(Tensor q, Tensor k, Tensor v, int heads, int length,
                                            CachedAttentionBackend.Options options) {
        return MetalCachedAttention.apply(graph, q, k, v, heads, length, options);
    }

    @Override
    public void writeHeadCache(Tensor cache, Tensor update, int heads, int offset) {
        HeadCacheOps.requireWrite(cache, update, heads, offset);
        if (!MetalDenseOps.matrix(cache) || !MetalDenseOps.matrix(update)) {
            HeadCacheOps.writeLayouts(this, cache, update, heads, offset);
            return;
        }
        cache.gradientState().beforeMutation();
        if (update.numel() == 0) return;
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
        if (!MetalDenseOps.matrix(left) || !MetalDenseOps.matrix(cache)) {
            return HeadCacheOps.product(this, left, cache, heads, length, transposeCache);
        }
        int columns = transposeCache ? length : cache.matrixCols();
        GpuBuffer output = graph.newOutputBuffer(left.matrixRows(), columns);
        graph.recordHeadPrefixMatmul(gpuIn(left), gpuIn(cache), output, heads, length, transposeCache,
                left.matrixRows(), left.matrixCols(), cache.matrixCols());
        return gpuOut(output);
    }
}
