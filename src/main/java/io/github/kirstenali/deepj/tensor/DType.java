package io.github.kirstenali.deepj.tensor;

import java.util.Objects;

/** Dense floating-point storage formats. Existing tensor factories default to FLOAT32. */
public enum DType {
    FLOAT32(4),
    FLOAT16(2),
    BFLOAT16(2);

    private final int bytes;

    DType(int bytes) { this.bytes = bytes; }

    public int bytes() { return bytes; }

    public long storageBytes(int elements) {
        if (elements < 0) throw new IllegalArgumentException("Element count cannot be negative");
        return (long) elements * bytes;
    }

    /** Different formats promote to FLOAT32; identical formats retain their dtype. */
    public static DType promote(DType left, DType right) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        return left == right ? left : FLOAT32;
    }

    public DType accumulationType() { return FLOAT32; }

    public DType gradientType() { return FLOAT32; }

    short encode16(float value) {
        return switch (this) {
            case FLOAT16 -> Float.floatToFloat16(value);
            case BFLOAT16 -> BFloat16.encode(value);
            case FLOAT32 -> throw new IllegalStateException("FLOAT32 does not use 16-bit storage");
        };
    }

    float decode16(short bits) {
        return switch (this) {
            case FLOAT16 -> Float.float16ToFloat(bits);
            case BFLOAT16 -> BFloat16.decode(bits);
            case FLOAT32 -> throw new IllegalStateException("FLOAT32 does not use 16-bit storage");
        };
    }
}
