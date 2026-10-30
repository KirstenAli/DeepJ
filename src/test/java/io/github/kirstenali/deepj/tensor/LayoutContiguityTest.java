package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

final class LayoutContiguityTest {
    @Test
    void cachedContiguityMatchesIndependentElementAddressesForRanksZeroThroughSix() {
        var random = new Random(2026);
        for (int rank = 0; rank <= 6; rank++) {
            for (int sample = 0; sample < 100; sample++) {
                int[] dimensions = random.ints(rank, 0, 4).toArray();
                long[] strides = random.longs(rank, 0, 10).toArray();
                verify(dimensions, strides, random.nextInt(4));
            }
        }
    }

    private static void verify(int[] dimensions, long[] strides, int offset) {
        TensorShape shape = new TensorShape(dimensions);
        int storage = storageSpan(dimensions, strides, offset);
        TensorLayout layout = new TensorLayout(shape, strides, offset, storage);
        boolean expected = consecutiveAddresses(layout, dimensions, strides);
        assertEquals(expected, layout.contiguous());
        assertTrue(TensorLayout.dense(shape).contiguous());
        if (dimensions.length > 0) dimensions[0] = 99;
        if (strides.length > 0) strides[0] = 999;
        assertEquals(expected, layout.contiguous());
    }

    private static int storageSpan(int[] dimensions, long[] strides, int offset) {
        long last = offset;
        for (int axis = 0; axis < dimensions.length; axis++) {
            last += strides[axis] * Math.max(0, dimensions[axis] - 1);
        }
        return Math.toIntExact(last + 1);
    }

    private static boolean consecutiveAddresses(TensorLayout layout, int[] dimensions, long[] strides) {
        for (int index = 0; index < layout.shape().elements(); index++) {
            int remainder = index;
            long address = layout.offset();
            for (int axis = dimensions.length - 1; axis >= 0; axis--) {
                address += (remainder % dimensions[axis]) * strides[axis];
                remainder /= dimensions[axis];
            }
            if (address != layout.offset() + index) return false;
        }
        return true;
    }
}
