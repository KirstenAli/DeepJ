package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

final class DTypeTest {
    @Test
    void formatsHaveExplicitPhysicalWidths() {
        assertEquals(4, DType.FLOAT32.bytes());
        assertEquals(2, DType.FLOAT16.bytes());
        assertEquals(2, DType.BFLOAT16.bytes());
        assertEquals(8_589_934_588L, DType.FLOAT32.storageBytes(Integer.MAX_VALUE));
        assertEquals(0, DType.FLOAT16.storageBytes(0));
        assertThrows(IllegalArgumentException.class, () -> DType.FLOAT32.storageBytes(-1));
    }

    @Test
    void promotionAndAccumulationDoNotMixTheTwoCompactFormats() {
        for (DType left : DType.values()) {
            assertEquals(DType.FLOAT32, left.accumulationType());
            assertEquals(DType.FLOAT32, left.gradientType());
            for (DType right : DType.values()) {
                assertEquals(left == right ? left : DType.FLOAT32, DType.promote(left, right));
            }
        }
        assertThrows(NullPointerException.class, () -> DType.promote(null, DType.FLOAT32));
        assertThrows(NullPointerException.class, () -> DType.promote(DType.FLOAT32, null));
    }

    @Test
    void float32DoesNotPretendToEncodeSixteenBitValues() {
        assertThrows(IllegalStateException.class, () -> DType.FLOAT32.encode16(1));
        assertThrows(IllegalStateException.class, () -> DType.FLOAT32.decode16((short) 0));
    }
}
