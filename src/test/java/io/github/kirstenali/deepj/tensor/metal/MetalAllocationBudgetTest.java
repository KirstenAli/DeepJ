package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;

class MetalAllocationBudgetTest {
    private static final Runnable NO_OP = () -> {};

    @Test
    void runtimeCanSelectTheDirectPathOnlyForDisabledBudgets() {
        assertFalse(new MetalAllocationBudget(0).enabled());
        assertTrue(new MetalAllocationBudget(1).enabled());
        assertTrue(new MetalAllocationBudget(Long.MAX_VALUE).enabled());
    }

    @Test
    void oversizedBatchIsRejectedBeforeAnyNativeAllocation() {
        var budget = new MetalAllocationBudget(32);
        allocate(budget, 1, 4);
        var called = new AtomicBoolean();
        var failure = assertThrows(IllegalStateException.class, () -> budget.allocate(
                new int[]{2, 3}, new int[]{4, 1}, null, 2, () -> called.set(true), NO_OP));
        assertTrue(failure.getMessage().contains("budget exceeded"));
        assertFalse(called.get());
        assertEquals(16, budget.liveBytes());
    }

    @Test
    void compactPhysicalBytesAndCrossGraphBatchesShareOneBudget() {
        var budget = new MetalAllocationBudget(32);
        budget.allocate(new int[]{1, 2}, new int[]{4, 4},
                new DType[]{DType.FLOAT16, DType.BFLOAT16}, 2, NO_OP, NO_OP);
        allocate(budget, 3, 4);
        assertEquals(32, budget.peakBytes());
        budget.release(new int[]{1, 1, 2, 99}, 4, NO_OP);
        assertEquals(16, budget.liveBytes());
        allocate(budget, 1, 4);
        assertEquals(32, budget.liveBytes());
    }

    @Test
    void duplicatesAndOverflowSizedRequestsFailWithoutChangingTheLedger() {
        var budget = new MetalAllocationBudget(32);
        allocate(budget, 1, 2);
        assertThrows(IllegalArgumentException.class, () -> allocate(budget, 1, 2));
        assertThrows(IllegalArgumentException.class, () -> budget.allocate(
                new int[]{2, 2}, new int[]{1, 1}, null, 2, NO_OP, NO_OP));
        assertThrows(IllegalStateException.class, () -> allocate(budget, 2, Integer.MAX_VALUE));
        assertEquals(8, budget.liveBytes());
        assertThrows(IllegalArgumentException.class, () -> new MetalAllocationBudget(-1));
    }

    @Test
    void partialNativeFailureRollsBackAndPreservesTheOriginalException() {
        var budget = new MetalAllocationBudget(32);
        var failure = new OutOfMemoryError("native allocation failed");
        var rolledBack = new AtomicBoolean();
        assertSame(failure, assertThrows(OutOfMemoryError.class, () -> budget.allocate(
                new int[]{1}, new int[]{4}, null, 1, () -> { throw failure; }, () -> rolledBack.set(true))));
        assertTrue(rolledBack.get());
        assertEquals(0, budget.liveBytes());
        assertEquals(0, budget.peakBytes());
    }

    @Test
    void failedRollbackIsSuppressedAndPreventsFurtherAllocations() {
        var budget = new MetalAllocationBudget(32);
        var failure = new IllegalStateException("allocate");
        var cleanup = new IllegalStateException("cleanup");
        assertSame(failure, assertThrows(IllegalStateException.class, () -> budget.allocate(
                new int[]{1}, new int[]{4}, null, 1, () -> { throw failure; }, () -> { throw cleanup; })));
        assertArrayEquals(new Throwable[]{cleanup}, failure.getSuppressed());
        assertThrows(IllegalStateException.class, () -> allocate(budget, 1, 4));
    }

    @Test
    void failedReleaseKeepsTheReservation() {
        var budget = new MetalAllocationBudget(32);
        allocate(budget, 1, 4);
        assertThrows(IllegalStateException.class, () -> budget.release(new int[]{1}, 1, () -> {
            throw new IllegalStateException("release failed");
        }));
        assertEquals(16, budget.liveBytes());
    }

    @Test
    void defaultDisabledBudgetDoesNotKeepAnAllocationLedger() {
        var budget = new MetalAllocationBudget(0);
        var called = new AtomicBoolean();
        budget.allocate(new int[]{1}, new int[]{Integer.MAX_VALUE}, null, 1, () -> called.set(true), NO_OP);
        assertTrue(called.get());
        budget.release(new int[]{1}, 1, NO_OP);
        assertEquals(0, budget.liveBytes());
        assertEquals(0, budget.peakBytes());
        assertNotNull(MetalAllocationBudget.configured());
    }

    private static void allocate(MetalAllocationBudget budget, int id, int elements) {
        budget.allocate(new int[]{id}, new int[]{elements}, null, 1, NO_OP, NO_OP);
    }
}
