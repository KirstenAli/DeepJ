package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;

class MetalAllocationBatchTest {
    private static final Runnable NO_OP = () -> {};

    @Test
    void nativeCallbackCannotChangeTheCapturedReservation() {
        var budget = new MetalAllocationBudget(32);
        int[] ids = {8, 3, 99};
        int[] sizes = {2, 4, 99};
        DType[] types = {DType.FLOAT32, DType.FLOAT16, DType.BFLOAT16};
        budget.allocate(ids, sizes, types, 2, () -> {
            ids[0] = 77;
            sizes[1] = 1;
            types[0] = DType.BFLOAT16;
        }, NO_OP);
        assertEquals(16, budget.liveBytes());
        budget.release(new int[]{8}, 1, NO_OP);
        assertEquals(8, budget.liveBytes());
        budget.release(new int[]{3}, 1, NO_OP);
        assertEquals(0, budget.liveBytes());
    }

    @Test
    void rejectedUnsortedBatchDoesNotPartlyReserveOtherIds() {
        var budget = new MetalAllocationBudget(32);
        var called = new AtomicBoolean();
        assertThrows(IllegalArgumentException.class, () -> budget.allocate(
                new int[]{9, 2, 6, 2}, new int[]{1, 1, 1, 1}, null, 4, () -> called.set(true), NO_OP));
        assertFalse(called.get());
        budget.allocate(new int[]{6, 2, 9}, new int[]{1, 1, 1}, null, 3, NO_OP, NO_OP);
        assertEquals(12, budget.liveBytes());
    }

    @Test
    void reservationPreservesIdSizePairsRatherThanSortedPositions() {
        var budget = new MetalAllocationBudget(Long.MAX_VALUE);
        budget.allocate(new int[]{81, 2, 15}, new int[]{7, 13, Integer.MAX_VALUE},
                new DType[]{DType.FLOAT16, DType.FLOAT32, DType.BFLOAT16}, 3, NO_OP, NO_OP);
        assertEquals(14L + 52 + 2L * Integer.MAX_VALUE, budget.liveBytes());
        budget.release(new int[]{2, 15}, 2, NO_OP);
        assertEquals(14, budget.liveBytes());
    }

    @Test
    void emptyBatchAndUnusedArrayTailDoNotReserveStorage() {
        var budget = new MetalAllocationBudget(4);
        budget.allocate(new int[]{1, 1}, new int[]{1, Integer.MAX_VALUE}, null, 0, NO_OP, NO_OP);
        assertEquals(0, budget.peakBytes());
        budget.allocate(new int[]{1, 1}, new int[]{1, Integer.MAX_VALUE}, null, 1, NO_OP, NO_OP);
        assertEquals(4, budget.liveBytes());
    }

    @Test
    void batchesMatchAnIndependentLedgerAcrossAllocationAndRelease() {
        var budget = new MetalAllocationBudget(Long.MAX_VALUE);
        Map<Integer, Long> expected = new HashMap<>();
        var random = new Random(8491);
        for (int round = 0; round < 200; round++) {
            allocateBatch(budget, expected, random, round);
            releaseSubset(budget, expected, random);
            long total = expected.values().stream().mapToLong(Long::longValue).sum();
            assertEquals(total, budget.liveBytes(), "round " + round);
        }
    }

    private static void allocateBatch(MetalAllocationBudget budget, Map<Integer, Long> expected,
                                      Random random, int round) {
        int count = 1 + random.nextInt(12);
        int[] ids = new int[count];
        int[] sizes = new int[count];
        DType[] types = new DType[count];
        for (int index = 0; index < count; index++) {
            ids[index] = 1000 + round * 16 + count - index;
            sizes[index] = 1 + random.nextInt(1000);
            types[index] = DType.values()[random.nextInt(3)];
            expected.put(ids[index], (long) sizes[index] * types[index].bytes());
        }
        budget.allocate(ids, sizes, types, count, NO_OP, NO_OP);
    }

    private static void releaseSubset(MetalAllocationBudget budget, Map<Integer, Long> expected, Random random) {
        int[] ids = expected.keySet().stream().filter(id -> random.nextBoolean()).mapToInt(Integer::intValue).toArray();
        budget.release(ids, ids.length, NO_OP);
        for (int id : ids) {
            expected.remove(id);
        }
    }
}
