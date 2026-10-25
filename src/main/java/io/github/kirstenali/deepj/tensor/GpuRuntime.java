package io.github.kirstenali.deepj.tensor;

public interface GpuRuntime {

    void allocBuffers(int[] ids, int[] sizes, int count);

    void uploadBuffer(int bufId, float[] data);

    /** Initializes the whole buffer with positive zero; custom runtimes retain a portable fallback. */
    default void zeroBuffer(int bufId, int floatCount) {
        if (bufId < 0 || floatCount < 1) throw new IllegalArgumentException("Invalid zero buffer");
        uploadBuffer(bufId, new float[floatCount]);
    }

    void downloadBuffer(int bufId, float[] out);

    void releaseBuffers(int[] ids, int count);

    void flushOps(int[] cmdStream, int cmdStreamLength);
}
