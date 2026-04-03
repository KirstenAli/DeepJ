package io.github.kirstenali.deepj.tensor;

/**
 * Handle to a GPU-resident float buffer managed by a {@link ComputeGraph}.
 *
 * <p>The buffer persists on the GPU across graph flushes until explicitly released.
 * Not tied to any specific GPU API (Metal, CUDA, etc.).
 */
public final class GpuBuffer {
    public final int id;
    public final int rows;
    public final int cols;

    /** True if this buffer was just allocated and needs CPU data uploaded before use. */
    public boolean needsUpload;

    /** True if GPU has newer data than Tensor.data (CPU cache is stale). */
    public boolean cpuStale;

    /** True if the runtime has allocated native memory for this id. */
    public boolean allocatedOnGpu;

    public GpuBuffer(int id, int rows, int cols, boolean needsUpload) {
        this.id = id;
        this.rows = rows;
        this.cols = cols;
        this.needsUpload = needsUpload;
        this.cpuStale = !needsUpload;
    }

    public int floatCount() { return rows * cols; }
}

