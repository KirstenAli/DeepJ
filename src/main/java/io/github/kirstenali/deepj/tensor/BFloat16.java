package io.github.kirstenali.deepj.tensor;

/** Round-to-nearest-even conversion; NaNs retain their high payload bits and are quieted. */
final class BFloat16 {
    private BFloat16() {}

    static short encode(float value) {
        int bits = Float.floatToRawIntBits(value);
        if (Float.isNaN(value)) return (short) ((bits >>> 16) | 0x0040);
        int rounding = 0x7fff + ((bits >>> 16) & 1);
        return (short) ((bits + rounding) >>> 16);
    }

    static float decode(short bits) {
        return Float.intBitsToFloat(Short.toUnsignedInt(bits) << 16);
    }
}
