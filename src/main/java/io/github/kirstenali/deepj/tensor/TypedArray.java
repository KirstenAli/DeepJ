package io.github.kirstenali.deepj.tensor;

import java.util.Objects;

/** Physical CPU allocation; compact formats have a short[] and no full-size float[] shadow. */
final class TypedArray {
    private final DType dtype;
    private final int length;
    private final Object values;

    TypedArray(DType dtype, int length) {
        this.dtype = Objects.requireNonNull(dtype, "dtype");
        dtype.storageBytes(length);
        this.length = length;
        values = dtype == DType.FLOAT32 ? new float[length] : new short[length];
    }

    DType dtype() { return dtype; }
    int length() { return length; }
    long byteLength() { return dtype.storageBytes(length); }

    float get(int index) {
        if (values instanceof float[] floats) return floats[index];
        return dtype.decode16(((short[]) values)[index]);
    }

    void set(int index, float value) {
        if (values instanceof float[] floats) floats[index] = value;
        else ((short[]) values)[index] = dtype.encode16(value);
    }

    float[] floats() {
        if (values instanceof float[] floats) return floats;
        throw new IllegalStateException("Compact storage has no float[] backing");
    }

    short[] shorts() {
        if (values instanceof short[] shorts) return shorts;
        throw new IllegalStateException("FLOAT32 storage has no short[] backing");
    }
}
