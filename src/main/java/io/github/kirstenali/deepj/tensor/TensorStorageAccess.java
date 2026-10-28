package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.GradientState;

/** Shared physical storage plumbing; logical Tensor operations remain on the concrete facade. */
public abstract class TensorStorageAccess {
    private final TensorMemory memory;
    private final TensorLayout layout;
    private final GradientState gradientState;
    private final int matrixRows, matrixCols;
    private final boolean flatStorage;
    private boolean deviceRetained;

    TensorStorageAccess(DType dtype, boolean host, int... dimensions) {
        this(TensorLayout.dense(new TensorShape(dimensions)), dtype, host);
    }

    TensorStorageAccess(DType dtype, boolean host, TensorLayout geometry) {
        this(geometry, dtype, host);
    }

    private TensorStorageAccess(TensorLayout layout, DType dtype, boolean host) {
        this(new TensorMemory(dtype, layout.shape().elements(), host), layout);
    }

    TensorStorageAccess(TensorMemory memory, TensorLayout layout) {
        this.memory = memory;
        this.layout = requireLayout(memory, layout);
        matrixRows = matrixDimension(layout, 0);
        matrixCols = matrixDimension(layout, 1);
        flatStorage = coversStorage(layout);
        gradientState = new GradientState();
        memory.bindRootState(gradientState);
    }

    TensorStorageAccess(TensorStorageAccess source, TensorLayout layout) {
        memory = source.memory;
        this.layout = requireLayout(memory, layout);
        matrixRows = matrixDimension(layout, 0);
        matrixCols = matrixDimension(layout, 1);
        flatStorage = coversStorage(layout);
        gradientState = memory.aliasState(source.gradientState);
    }

    final TensorMemory memory() { return memory; }
    final TensorLayout layout() { return layout; }
    final boolean coversStorage() { return flatStorage; }

    public final int rank() { return layout.shape().rank(); }
    public final int[] shape() { return layout.shape().dimensions(); }
    /** Tests exact logical dimensions without copying shape metadata or reading tensor values. */
    public final boolean hasShape(int... dimensions) { return layout.shape().matches(dimensions); }
    public final int size(int axis) { return layout.shape().size(axis); }
    public final int numel() { return layout.shape().elements(); }
    public final boolean hasSameShape(TensorStorageAccess other) { return layout.shape().equals(other.layout.shape()); }
    public final long[] strides() { return layout.strides(); }
    public final int storageOffset() { return layout.offset(); }
    public final boolean isContiguous() { return layout.contiguous(); }
    /** Physical bytes in the shared allocation; views may address only part of it. */
    public final long storageBytes() { return memory.byteLength(); }

    /** Checked rank-two dimensions for matrix kernels; other ranks are never implicitly flattened. */
    public final int matrixRows() { return requireMatrixDimension(matrixRows); }
    public final int matrixCols() { return requireMatrixDimension(matrixCols); }

    public abstract void materialize();

    public final float get(int row, int column) { return TensorValues.get(this, row, column); }
    public final float get(int... indices) { return TensorValues.get(this, indices); }

    public final void set(int row, int column, float value) { TensorValues.set(this, row, column, value); }
    public final void setAt(float value, int... indices) { TensorValues.set(this, value, indices); }

    /** Returns a logical row-major copy, converting compact storage to FLOAT32 values when necessary. */
    public final float[] toFloatArray() { return TensorValues.toFloatArray(this); }

    public final GradientState gradientState() { return gradientState; }

    public final Object getGpuTag() { return memory.gpuTag; }

    public final void setGpuTag(Object tag) { memory.gpuTag = tag; }

    public final DType dtype() { return memory.dtype(); }

    /** Unsafe backing access bypasses mutation tracking; use managed Tensor get/set for application code. */
    public final float[] rawData() {
        if (!flatStorage) throw new IllegalStateException("Raw storage requires a full contiguous tensor");
        return memory.rawFloats();
    }

    private static TensorLayout requireLayout(TensorMemory memory, TensorLayout layout) {
        if (memory.length() != layout.storageElements()) {
            throw new IllegalArgumentException("Tensor layout must address its own physical storage");
        }
        return layout;
    }

    private static boolean coversStorage(TensorLayout layout) {
        return layout.offset() == 0 && layout.shape().elements() == layout.storageElements() && layout.contiguous();
    }

    private static int matrixDimension(TensorLayout layout, int axis) {
        return layout.shape().rank() == 2 ? layout.shape().size(axis) : -1;
    }

    private static int requireMatrixDimension(int dimension) {
        if (dimension < 0) throw new IllegalArgumentException("Matrix operation requires a rank-two tensor");
        return dimension;
    }

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
