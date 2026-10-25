package io.github.kirstenali.deepj.tensor;

/** Arithmetic IEEE binary16 reference, independent of the JDK conversion used in production. */
final class CompactFloatReference {
    private CompactFloatReference() {}

    static float half(int bits) {
        int exponent = (bits >>> 10) & 31, fraction = bits & 1023;
        float sign = (bits & 0x8000) == 0 ? 1 : -1;
        if (exponent == 31) return fraction == 0 ? sign * Float.POSITIVE_INFINITY : Float.NaN;
        if (exponent == 0) return Math.scalb(sign * fraction, -24);
        return Math.scalb(sign * (1024 + fraction), exponent - 25);
    }

    static short bfloat(float value) {
        int bits = Float.floatToRawIntBits(value), upper = bits >>> 16, lower = bits & 0xffff;
        if (Float.isNaN(value)) return (short) (upper | 0x40);
        if (lower > 0x8000 || (lower == 0x8000 && (upper & 1) != 0)) upper++;
        return (short) upper;
    }
}
