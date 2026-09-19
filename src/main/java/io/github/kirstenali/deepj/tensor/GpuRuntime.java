package io.github.kirstenali.deepj.tensor;

public interface GpuRuntime {

    void allocBuffers(int[] ids, int[] sizes, int count);

    void uploadBuffer(int bufId, float[] data);

    void downloadBuffer(int bufId, float[] out);

    void releaseBuffers(int[] ids, int count);

    void flushOps(int[] cmdStream, int cmdStreamLength);
}
