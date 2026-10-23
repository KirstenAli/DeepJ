package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;

abstract class MetalAttentionOps extends MetalTrainingOps implements CausalAttentionBackend {
    @Override
    public boolean supportsCausalAttention(Tensor query, Tensor key, Tensor value, int heads, float scale) {
        return MetalNative.TILED_ATTENTION_AVAILABLE && tiledShape(query, heads, scale)
                && query.rows / heads >= 256 && sameShape(query, key) && sameShape(query, value);
    }

    private static boolean tiledShape(Tensor query, int heads, float scale) {
        return heads > 0 && query.rows % heads == 0 && query.cols == 64
                && query.rows / heads >= 32 && query.rows / heads % 32 == 0 && Float.isFinite(scale);
    }

    private static boolean sameShape(Tensor left, Tensor right) {
        return left.rows == right.rows && left.cols == right.cols;
    }

    @Override
    public CausalAttentionResult causalAttention(Tensor query, Tensor key, Tensor value, int heads, float scale) {
        requireAttentionShapes(query, key, value, heads, scale);
        GpuBuffer output = graph.newOutputBuffer(query.rows, query.cols);
        GpuBuffer normalization = graph.newOutputBuffer(query.rows, 2);
        graph.recordCausalAttention(gpuIn(query), gpuIn(key), gpuIn(value), output, normalization, heads, scale);
        return new CausalAttentionResult(gpuOut(output), gpuOut(normalization));
    }

    @Override
    public AttentionGradients causalAttentionBackward(Tensor query, Tensor key, Tensor value,
                                                      CausalAttentionResult saved, Tensor gradient,
                                                      int heads, float scale) {
        requireAttentionShapes(query, key, value, heads, scale);
        Tensor.requireSameShape(query, gradient, "causalAttentionBackward");
        Tensor.requireSameShape(query, saved.output(), "causalAttentionBackward");
        requireNormalization(saved.normalization(), query.rows);
        GpuBuffer dq = graph.newOutputBuffer(query.rows, query.cols);
        GpuBuffer dk = graph.newOutputBuffer(query.rows, query.cols);
        GpuBuffer dv = graph.newOutputBuffer(query.rows, query.cols);
        GpuBuffer delta = graph.newOutputBuffer(query.rows, 1);
        graph.recordCausalAttentionBackward(gpuIn(query), gpuIn(key), gpuIn(value), gpuIn(saved.output()),
                gpuIn(gradient), gpuIn(saved.normalization()), delta, dq, dk, dv, heads, scale);
        return new AttentionGradients(gpuOut(dq), gpuOut(dk), gpuOut(dv));
    }

    private static void requireNormalization(Tensor normalization, int rows) {
        if (normalization.rows != rows || normalization.cols != 2) {
            throw new IllegalArgumentException("Causal attention normalization must contain one pair per row");
        }
    }

    private static void requireAttentionShapes(Tensor query, Tensor key, Tensor value, int heads, float scale) {
        Tensor.requireSameShape(query, key, "causalAttention");
        Tensor.requireSameShape(query, value, "causalAttention");
        if (!tiledShape(query, heads, scale)) {
            throw new IllegalArgumentException("Tiled attention needs complete 32-row tiles and 64-column heads");
        }
        if (!MetalNative.TILED_ATTENTION_AVAILABLE) {
            throw new UnsupportedOperationException("Tiled attention requires an Apple 7 family GPU or newer");
        }
    }
}
