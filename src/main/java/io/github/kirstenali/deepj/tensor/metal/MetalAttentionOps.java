package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;

abstract class MetalAttentionOps extends MetalCacheOps implements CausalAttentionBackend {
    @Override
    public boolean supportsCausalAttention(Tensor query, Tensor key, Tensor value, int heads, float scale) {
        return supportsCausalAttention(query, key, value, heads, heads, scale);
    }

    @Override
    public boolean supportsCausalAttention(Tensor query, Tensor key, Tensor value,
                                            int queryHeads, int keyValueHeads, float scale) {
        return MetalNative.TILED_ATTENTION_AVAILABLE && denseInputs(query, key, value)
                && tiledShape(query, queryHeads, scale)
                && query.matrixRows() / queryHeads >= 256 && groupedShape(query, key, queryHeads, keyValueHeads)
                && sameShape(key, value);
    }

    private static boolean denseInputs(Tensor query, Tensor key, Tensor value) {
        return MetalDenseOps.matrix(query) && MetalDenseOps.matrix(key) && MetalDenseOps.matrix(value);
    }

    private static boolean tiledShape(Tensor query, int heads, float scale) {
        return heads > 0 && query.matrixRows() % heads == 0 && query.matrixCols() == 64
                && query.matrixRows() / heads >= 32 && query.matrixRows() / heads % 32 == 0 && Float.isFinite(scale);
    }

    private static boolean sameShape(Tensor left, Tensor right) {
        return left.matrixRows() == right.matrixRows() && left.matrixCols() == right.matrixCols();
    }

    private static boolean groupedShape(Tensor query, Tensor key, int queryHeads, int keyValueHeads) {
        return queryHeads > 0 && keyValueHeads > 0 && queryHeads % keyValueHeads == 0
                && key.matrixRows() % keyValueHeads == 0 && query.matrixCols() == key.matrixCols()
                && query.matrixRows() / queryHeads == key.matrixRows() / keyValueHeads;
    }

    @Override
    public CausalAttentionResult causalAttention(Tensor query, Tensor key, Tensor value, int heads, float scale) {
        return causalAttention(query, key, value, heads, heads, scale);
    }

    @Override
    public CausalAttentionResult causalAttention(Tensor query, Tensor key, Tensor value,
                                                 int queryHeads, int keyValueHeads, float scale) {
        requireAttentionShapes(query, key, value, queryHeads, keyValueHeads, scale);
        GpuBuffer output = graph.newOutputBuffer(query.matrixRows(), query.matrixCols());
        GpuBuffer normalization = graph.newOutputBuffer(query.matrixRows(), 2);
        graph.recordCausalAttention(gpuIn(query), gpuIn(key), gpuIn(value), output, normalization,
                queryHeads, keyValueHeads, scale);
        return new CausalAttentionResult(graph.createOutputLike(output, query), gpuOut(normalization));
    }

    @Override
    public AttentionGradients causalAttentionBackward(Tensor query, Tensor key, Tensor value,
                                                      CausalAttentionResult saved, Tensor gradient,
                                                      int heads, float scale) {
        return causalAttentionBackward(query, key, value, saved, gradient, heads, heads, scale);
    }

    @Override
    public AttentionGradients causalAttentionBackward(Tensor query, Tensor key, Tensor value,
                                                      CausalAttentionResult saved, Tensor gradient,
                                                      int queryHeads, int keyValueHeads, float scale) {
        requireAttentionShapes(query, key, value, queryHeads, keyValueHeads, scale);
        Tensor.requireSameShape(query, gradient, "causalAttentionBackward");
        Tensor.requireSameShape(query, saved.output(), "causalAttentionBackward");
        requireNormalization(saved.normalization(), query.matrixRows());
        GpuBuffer dq = graph.newOutputBuffer(query.matrixRows(), query.matrixCols());
        GpuBuffer dk = graph.newOutputBuffer(key.matrixRows(), key.matrixCols());
        GpuBuffer dv = graph.newOutputBuffer(value.matrixRows(), value.matrixCols());
        GpuBuffer delta = graph.newOutputBuffer(query.matrixRows(), 1);
        graph.recordCausalAttentionBackward(gpuIn(query), gpuIn(key), gpuIn(value), gpuIn(saved.output()),
                gpuIn(gradient), gpuIn(saved.normalization()), delta, dq, dk, dv,
                queryHeads, keyValueHeads, scale);
        return new AttentionGradients(graph.createOutputLike(dq, query), graph.createOutputLike(dk, key),
                graph.createOutputLike(dv, value));
    }

    private static void requireNormalization(Tensor normalization, int rows) {
        if (normalization.matrixRows() != rows || normalization.matrixCols() != 2) {
            throw new IllegalArgumentException("Causal attention normalization must contain one pair per row");
        }
    }

    private static void requireAttentionShapes(Tensor query, Tensor key, Tensor value,
                                                int queryHeads, int keyValueHeads, float scale) {
        Tensor.requireSameShape(key, value, "causalAttention");
        if (!denseInputs(query, key, value)) {
            throw new UnsupportedOperationException("Fused attention requires full contiguous FLOAT32 matrices");
        }
        if (!tiledShape(query, queryHeads, scale) || !groupedShape(query, key, queryHeads, keyValueHeads)) {
            throw new IllegalArgumentException("Tiled attention needs complete 32-row tiles and 64-column heads");
        }
        if (!MetalNative.TILED_ATTENTION_AVAILABLE) {
            throw new UnsupportedOperationException("Block attention is not available on this Metal device");
        }
    }
}
