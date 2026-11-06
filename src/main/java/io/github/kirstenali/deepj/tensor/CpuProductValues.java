package io.github.kirstenali.deepj.tensor;

/** A call-local view of existing physical storage; it never retains converted weights. */
final class CpuProductValues {
    private final float[] floats;
    private final short[] shorts;
    private final boolean half;
    private final int type;

    CpuProductValues(Tensor tensor) {
        floats = tensor.dtype() == DType.FLOAT32 ? tensor.memory().floats() : null;
        shorts = floats == null ? tensor.memory().shorts() : null;
        half = tensor.dtype() == DType.FLOAT16;
        type = tensor.dtype().storageCode();
    }

    float get(int address) {
        if (floats != null) return floats[address];
        short value = shorts[address];
        return half ? Float.float16ToFloat(value) : BFloat16.decode(value);
    }

    float[] floats() { return floats; }
    Object storage() { return floats == null ? shorts : floats; }
    int type() { return type; }
}
