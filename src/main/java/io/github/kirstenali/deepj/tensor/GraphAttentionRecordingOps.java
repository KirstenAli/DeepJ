package io.github.kirstenali.deepj.tensor;

abstract class GraphAttentionRecordingOps extends GraphAdvancedRecordingOps {
    GraphAttentionRecordingOps(GpuRuntime runtime) { super(runtime); }

    public void recordCausalAttention(GpuBuffer query, GpuBuffer key, GpuBuffer value,
                                      GpuBuffer output, GpuBuffer normalization, int heads, float scale) {
        beginOp(9);
        emitInt(OP_TILED_CAUSAL_ATTENTION);
        emitBuffers(query, key, value, output, normalization);
        emitInt(query.rows);
        emitInt(heads);
        emitFloatBits(scale);
        endOp();
    }

    public void recordCausalAttentionBackward(GpuBuffer query, GpuBuffer key, GpuBuffer value,
                                              GpuBuffer output, GpuBuffer gradient, GpuBuffer normalization,
                                              GpuBuffer delta, GpuBuffer dq, GpuBuffer dk, GpuBuffer dv,
                                              int heads, float scale) {
        beginOp(14);
        emitInt(OP_TILED_CAUSAL_ATTENTION_BACKWARD);
        emitBuffers(query, key, value, output, gradient, normalization, delta, dq, dk, dv);
        emitInt(query.rows);
        emitInt(heads);
        emitFloatBits(scale);
        endOp();
    }

    private void emitBuffers(GpuBuffer... buffers) {
        for (GpuBuffer buffer : buffers) {
            emitInt(buffer.id);
        }
    }
}
