package io.github.kirstenali.deepj.tensor;

/** Transfers a complete shared allocation, never a misleading flattened view. */
final class GraphTransfers {
    private GraphTransfers() {}

    static Object snapshot(Tensor tensor) {
        tensor.materialize();
        TensorMemory memory = tensor.memory();
        return memory.dtype() == DType.FLOAT32 ? memory.floats().clone() : memory.shorts().clone();
    }

    static void upload(GpuRuntime runtime, int[] upload, Object data) {
        if (data instanceof float[] values) runtime.uploadBuffer(upload[0], values);
        else if (data instanceof short[] values) runtime.uploadBuffer16(upload[0], values);
        else runtime.zeroBuffer(upload[0], upload[1], GpuAllocations.type(upload));
    }

    static void download(GpuRuntime runtime, GpuBuffer buffer, TensorMemory memory) {
        if (memory.dtype() == DType.FLOAT32) runtime.downloadBuffer(buffer.id, memory.floats());
        else runtime.downloadBuffer16(buffer.id, memory.shorts());
        buffer.cpuStale = false;
        buffer.needsUpload = false;
    }
}
