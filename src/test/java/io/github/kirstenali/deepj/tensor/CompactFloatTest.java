package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

final class CompactFloatTest {
    @Test
    void allBinary16PatternsMatchTheIndependentArithmeticReference() {
        for (int bits = 0; bits <= 0xffff; bits++) {
            float expected = CompactFloatReference.half(bits), actual = DType.FLOAT16.decode16((short) bits);
            if (Float.isNaN(expected)) assertTrue(Float.isNaN(actual));
            else assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(actual));
        }
    }

    @Test
    void halfSpecialValuesAndRoundingTiesHaveKnownEncodings() {
        assertEquals((short) 0x3e00, DType.FLOAT16.encode16(1.5f));
        assertEquals((short) 0x8000, DType.FLOAT16.encode16(-0.0f));
        assertEquals((short) 0x0001, DType.FLOAT16.encode16(Math.scalb(1.0f, -24)));
        assertEquals((short) 0x0400, DType.FLOAT16.encode16(Math.scalb(1.0f, -14)));
        assertEquals((short) 0x0000, DType.FLOAT16.encode16(Math.scalb(1.0f, -25)));
        assertEquals((short) 0x3c00, DType.FLOAT16.encode16(1.00048828125f));
        assertEquals((short) 0x3c02, DType.FLOAT16.encode16(1.00146484375f));
        assertEquals((short) 0x7bff, DType.FLOAT16.encode16(65504));
        assertEquals((short) 0x7c00, DType.FLOAT16.encode16(65520));
        assertEquals((short) 0xfc00, DType.FLOAT16.encode16(Float.NEGATIVE_INFINITY));
        assertTrue(Float.isNaN(DType.FLOAT16.decode16(DType.FLOAT16.encode16(Float.NaN))));
    }

    @Test
    void bfloatRoundingMatchesIndependentBranchesForRandomFloatBits() {
        var random = new Random(2026);
        for (int index = 0; index < 100_000; index++) {
            float value = Float.intBitsToFloat(random.nextInt());
            assertEquals(CompactFloatReference.bfloat(value), DType.BFLOAT16.encode16(value));
        }
    }

    @Test
    void bfloatSpecialValuesAndRoundingTiesHaveKnownEncodings() {
        assertEquals((short) 0x3fc0, DType.BFLOAT16.encode16(1.5f));
        assertEquals((short) 0x8000, DType.BFLOAT16.encode16(-0.0f));
        assertEquals((short) 0x0001, DType.BFLOAT16.encode16(Float.intBitsToFloat(0x00010000)));
        assertEquals((short) 0x0000, DType.BFLOAT16.encode16(Float.intBitsToFloat(0x00008000)));
        assertEquals((short) 0x3f80, DType.BFLOAT16.encode16(1.00390625f));
        assertEquals((short) 0x3f82, DType.BFLOAT16.encode16(1.01171875f));
        assertEquals((short) 0x7f80, DType.BFLOAT16.encode16(Float.MAX_VALUE));
        assertEquals((short) 0xff80, DType.BFLOAT16.encode16(Float.NEGATIVE_INFINITY));
        assertTrue(Float.isNaN(DType.BFLOAT16.decode16(DType.BFLOAT16.encode16(Float.intBitsToFloat(0x7f800001)))));
    }

    @Test
    void everyBfloatStoragePatternDecodesWithoutLosingRawBits() {
        for (int bits = 0; bits <= 0xffff; bits++) {
            float expected = Float.intBitsToFloat(bits << 16), actual = DType.BFLOAT16.decode16((short) bits);
            assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(actual));
        }
    }
}
