package io.github.kirstenali.deepj.tensor;

abstract class GraphAttentionRecordingOps extends GraphAdvancedRecordingOps {
    GraphAttentionRecordingOps(GpuRuntime runtime) { super(runtime); }

    public void recordHeadCacheWrite(GpuBuffer update, GpuBuffer cache, int heads, int offset) {
        beginOp(8);
        emitInt(OP_HEAD_CACHE_WRITE);
        emitBuffers(update, cache);
        emitInt(heads);
        emitInt(update.rows / heads);
        emitInt(cache.rows / heads);
        emitInt(cache.cols);
        emitInt(offset);
        endOp();
    }

    public void recordHeadPrefixMatmul(GpuBuffer left, GpuBuffer cache, GpuBuffer output,
                                       int heads, int length, boolean transposeCache) {
        beginOp(10);
        emitInt(OP_HEAD_PREFIX_MATMUL);
        emitBuffers(left, cache, output);
        emitInt(heads);
        emitInt(left.rows / heads);
        emitInt(left.cols);
        emitInt(length);
        emitInt(cache.cols);
        emitInt(transposeCache ? 1 : 0);
        endOp();
    }

    public void recordCausalAttention(GpuBuffer query, GpuBuffer key, GpuBuffer value,
                                      GpuBuffer output, GpuBuffer normalization, int heads, float scale) {
        recordCausalAttention(query, key, value, output, normalization, heads, heads, scale);
    }

    public void recordCausalAttention(GpuBuffer query, GpuBuffer key, GpuBuffer value,
                                      GpuBuffer output, GpuBuffer normalization,
                                      int queryHeads, int keyValueHeads, float scale) {
        beginOp(10);
        emitInt(OP_TILED_CAUSAL_ATTENTION);
        emitBuffers(query, key, value, output, normalization);
        emitInt(query.rows);
        emitInt(queryHeads);
        emitInt(keyValueHeads);
        emitFloatBits(scale);
        endOp();
    }

    public void recordCausalAttentionBackward(GpuBuffer query, GpuBuffer key, GpuBuffer value,
                                              GpuBuffer output, GpuBuffer gradient, GpuBuffer normalization,
                                              GpuBuffer delta, GpuBuffer dq, GpuBuffer dk, GpuBuffer dv,
                                              int heads, float scale) {
        recordCausalAttentionBackward(query, key, value, output, gradient, normalization,
                delta, dq, dk, dv, heads, heads, scale);
    }

    public void recordCausalAttentionBackward(GpuBuffer query, GpuBuffer key, GpuBuffer value,
                                              GpuBuffer output, GpuBuffer gradient, GpuBuffer normalization,
                                              GpuBuffer delta, GpuBuffer dq, GpuBuffer dk, GpuBuffer dv,
                                              int queryHeads, int keyValueHeads, float scale) {
        beginOp(15);
        emitInt(OP_TILED_CAUSAL_ATTENTION_BACKWARD);
        emitBuffers(query, key, value, output, gradient, normalization, delta, dq, dk, dv);
        emitInt(query.rows);
        emitInt(queryHeads);
        emitInt(keyValueHeads);
        emitFloatBits(scale);
        endOp();
    }

    private void emitBuffers(GpuBuffer... buffers) {
        for (GpuBuffer buffer : buffers) {
            emitBuffer(buffer);
        }
    }
}
