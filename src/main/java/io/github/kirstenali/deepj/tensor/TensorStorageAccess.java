package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.GradientState;

/** Shared physical storage plumbing; logical Tensor operations remain on the concrete facade. */
public abstract class TensorStorageAccess {
    private final TensorMemory memory;
    private final GradientState gradientState;
    private boolean deviceRetained;

    TensorStorageAccess(TensorMemory memory) {
        this.memory = memory;
        gradientState = new GradientState();
        memory.bindRootState(gradientState);
    }

    TensorStorageAccess(TensorStorageAccess source) {
        memory = source.memory;
        gradientState = memory.aliasState(source.gradientState);
    }

    final TensorMemory memory() { return memory; }

    public final GradientState gradientState() { return gradientState; }

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
