package io.github.kirstenali.deepj.tensor;

public final class GpuBuffer {
    public final int id;
    public final int rows;
    public final int cols;

    public boolean needsUpload;

    public boolean cpuStale;

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
