package io.github.kirstenali.deepj.tensor;

import java.util.Objects;

/** Dense floating-point storage formats. Existing tensor factories default to FLOAT32. */
public enum DType {
    FLOAT32(1, 4),
    FLOAT16(2, 2),
    BFLOAT16(3, 2);

    private final int storageCode;
    private final int bytes;

    DType(int storageCode, int bytes) {
        this.storageCode = storageCode;
        this.bytes = bytes;
    }

    public int bytes() { return bytes; }
    /** Stable storage identifier shared by checkpoint and native buffer descriptors. */
    public int storageCode() { return storageCode; }

    public static DType fromStorageCode(int code) {
        return switch (code) {
            case 1 -> FLOAT32;
            case 2 -> FLOAT16;
            case 3 -> BFLOAT16;
            default -> throw new IllegalArgumentException("Unknown tensor dtype: " + code);
        };
    }

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
