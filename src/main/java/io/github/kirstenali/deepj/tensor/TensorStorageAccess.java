package io.github.kirstenali.deepj.tensor;

/** Shared physical storage plumbing; logical Tensor operations remain on the concrete facade. */
abstract class TensorStorageAccess {
    private final TensorMemory memory;
    private boolean deviceRetained;

    TensorStorageAccess(TensorMemory memory) { this.memory = memory; }

    final TensorMemory memory() { return memory; }

    public final Object getGpuTag() { return memory.gpuTag; }

    public final void setGpuTag(Object tag) { memory.gpuTag = tag; }

    public final DType dtype() { return memory.dtype(); }

    /** Unsafe backing access bypasses mutation tracking; use managed Tensor get/set for application code. */
    public final float[] rawData() { return memory.rawFloats(); }

    final void retainDevice() {
        if (!deviceRetained) memory.retainDevice();
        deviceRetained = true;
    }

    public final boolean retainsDeviceBuffer() { return memory.deviceRetained(); }

    public final void releaseDeviceBufferRetention() {
        if (deviceRetained) memory.releaseDevice();
        deviceRetained = false;
    }
}
