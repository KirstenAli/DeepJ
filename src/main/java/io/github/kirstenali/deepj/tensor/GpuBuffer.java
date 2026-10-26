package io.github.kirstenali.deepj.tensor;

import java.util.Objects;

public final class GpuBuffer {
    GraphExecution owner;
    private volatile boolean released;
    public final int id;
    public final int rows;
    public final int cols;
    public final DType dtype;

    public boolean needsUpload;

    public boolean cpuStale;

    public boolean allocatedOnGpu;

    public GpuBuffer(int id, int rows, int cols, boolean needsUpload) {
        this(id, rows, cols, DType.FLOAT32, needsUpload);
    }

    public GpuBuffer(int id, int rows, int cols, DType dtype, boolean needsUpload) {
        if (id < 0) throw new IllegalArgumentException("GPU buffer id cannot be negative");
        TensorStorage.checkedSize(rows, cols);
        this.id = id;
        this.rows = rows;
        this.cols = cols;
        this.dtype = Objects.requireNonNull(dtype, "dtype");
        this.needsUpload = needsUpload;
        this.cpuStale = !needsUpload;
    }

    public int elementCount() { return rows * cols; }
    public long byteLength() { return dtype.storageBytes(elementCount()); }

    public int floatCount() {
        requireFloat32();
        return elementCount();
    }

    void requireFloat32() {
        if (dtype != DType.FLOAT32) throw new UnsupportedOperationException("Matrix opcode requires FLOAT32");
    }

    void requireLive() {
        if (released) throw new IllegalStateException("GPU buffer has been released");
    }

    void markReleased() {
        released = true;
        allocatedOnGpu = false;
    }
}
