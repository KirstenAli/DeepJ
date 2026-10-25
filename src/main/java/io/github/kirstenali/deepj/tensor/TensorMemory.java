package io.github.kirstenali.deepj.tensor;

import java.util.Objects;

/** Physical typed allocation and device binding, independent of any logical tensor layout. */
final class TensorMemory {
    private final DType dtype;
    private final int length;
    private final boolean lazyHost;
    private Object values;
    private int deviceRetentions;
    Object gpuTag;

    TensorMemory(DType dtype, int length) { this(dtype, length, true); }

    TensorMemory(DType dtype, int length, boolean host) {
        this.dtype = Objects.requireNonNull(dtype, "dtype");
        dtype.storageBytes(length);
        this.length = length;
        lazyHost = !host;
        if (host) ensureHost();
    }

    DType dtype() { return dtype; }
    int length() { return length; }
    long byteLength() { return dtype.storageBytes(length); }

    float get(int index) {
        ensureHost();
        if (values instanceof float[] floats) return floats[index];
        return dtype.decode16(((short[]) values)[index]);
    }

    void set(int index, float value) {
        ensureHost();
        if (values instanceof float[] floats) floats[index] = value;
        else ((short[]) values)[index] = dtype.encode16(value);
    }

    float[] floats() {
        ensureHost();
        if (values instanceof float[] floats) return floats;
        throw new IllegalStateException("Compact storage has no float[] backing");
    }

    short[] shorts() {
        ensureHost();
        if (values instanceof short[] shorts) return shorts;
        throw new IllegalStateException("FLOAT32 storage has no short[] backing");
    }

    float[] rawFloats() {
        if (dtype != DType.FLOAT32) throw new IllegalStateException("Compact storage has no float[] backing");
        return (float[]) values;
    }

    boolean hasHostStorage() { return values != null; }

    void discardStaleHost() {
        if (lazyHost) values = null;
    }

    private void ensureHost() {
        if (values == null) values = dtype == DType.FLOAT32 ? new float[length] : new short[length];
    }

    TensorMemory copy() {
        ensureHost();
        var result = new TensorMemory(dtype, length);
        System.arraycopy(values, 0, result.values, 0, length);
        return result;
    }

    void retainDevice() { deviceRetentions = Math.incrementExact(deviceRetentions); }

    boolean deviceRetained() { return deviceRetentions > 0; }

    void releaseDevice() {
        if (deviceRetentions == 0) throw new IllegalStateException("Unbalanced device retention");
        deviceRetentions--;
    }
}
