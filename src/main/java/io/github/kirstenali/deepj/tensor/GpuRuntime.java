package io.github.kirstenali.deepj.tensor;

public interface GpuRuntime {

    default boolean supportsLayouts() { return false; }

    void allocBuffers(int[] ids, int[] sizes, int count);

    /** Physical element counts and dtypes, independent of logical Tensor shape. */
    default void allocTypedBuffers(int[] ids, int[] sizes, DType[] dtypes, int count) {
        GpuAllocations.requireValid(ids, sizes, dtypes, count);
        if (!GpuAllocations.allFloat32(dtypes, count)) {
            throw new UnsupportedOperationException("Runtime does not support compact GPU storage");
        }
        allocBuffers(ids, sizes, count);
    }

    void uploadBuffer(int bufId, float[] data);

    default void uploadBuffer16(int bufId, short[] data) {
        throw new UnsupportedOperationException("Runtime does not support 16-bit uploads");
    }

    /** Initializes the whole buffer with positive zero; custom runtimes retain a portable fallback. */
    default void zeroBuffer(int bufId, int floatCount) {
        if (bufId < 0 || floatCount < 1) throw new IllegalArgumentException("Invalid zero buffer");
        uploadBuffer(bufId, new float[floatCount]);
    }

    void downloadBuffer(int bufId, float[] out);

    default void downloadBuffer16(int bufId, short[] out) {
        throw new UnsupportedOperationException("Runtime does not support 16-bit downloads");
    }

    default void zeroBuffer(int bufId, int elements, DType dtype) {
        java.util.Objects.requireNonNull(dtype, "dtype");
        if (bufId < 0 || elements < 1) throw new IllegalArgumentException("Invalid zero buffer");
        if (dtype == DType.FLOAT32) zeroBuffer(bufId, elements);
        else uploadBuffer16(bufId, new short[elements]);
    }

    void releaseBuffers(int[] ids, int count);

    void flushOps(int[] cmdStream, int cmdStreamLength);

    default void flushOps(int[] commands, int length, long[] layouts) {
        if (java.util.Objects.requireNonNull(layouts).length != 0) {
            throw new UnsupportedOperationException("Runtime does not support GPU layout descriptors");
        }
        flushOps(commands, length);
    }
}
