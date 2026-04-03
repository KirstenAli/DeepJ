package io.github.kirstenali.deepj.tensor;

/**
 * Abstraction over a GPU compute runtime (Metal, CUDA, Vulkan, etc.).
 *
 * <p>{@link ComputeGraph} delegates all native/driver calls through this interface,
 * keeping the graph logic itself backend-agnostic.
 */
public interface GpuRuntime {

    /** Batch-allocate GPU buffers. {@code ids[i]} gets {@code sizes[i]} floats. */
    void allocBuffers(int[] ids, int[] sizes, int count);

    /** Upload CPU float data into an existing GPU buffer. */
    void uploadBuffer(int bufId, float[] data);

    /** Download GPU buffer contents into a CPU float array. */
    void downloadBuffer(int bufId, float[] out);

    /** Release multiple GPU buffers by id. */
    void releaseBuffers(int[] ids, int count);

    /**
     * Execute a batch of ops encoded as a flat {@code int[]} command stream,
     * all in one command buffer submission.
     */
    void flushOps(int[] cmdStream, int cmdStreamLength);
}

