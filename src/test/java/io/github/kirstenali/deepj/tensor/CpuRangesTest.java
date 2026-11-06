package io.github.kirstenali.deepj.tensor;

import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CpuRangesTest {
    @Test void fatalWorkerFailuresReachTheCallerInsteadOfReturningPartialOutput() {
        var failure = new AssertionError("simulated native allocation failure");
        Error actual = assertThrows(AssertionError.class, () -> CpuRanges.forRange(0, 100, index -> {
            if (index == 70) throw failure;
        }));
        assertSame(failure, actual);
    }

    @Test void singleWorkItemRunsOnTheCallerAndEmptyRangesDoNothing() {
        Thread caller = Thread.currentThread();
        CpuRanges.forRange(3, 4, index -> assertSame(caller, Thread.currentThread()));
        var count = new AtomicInteger();
        CpuRanges.forRange(4, 4, index -> count.incrementAndGet());
        assertEquals(0, count.get());
    }

    @Test void balancedTilesAreVisitedExactlyOnceAndWorkerErrorsReachTheCaller() {
        var count = new java.util.concurrent.atomic.AtomicIntegerArray(37);
        CpuRanges.balance(37, index -> count.incrementAndGet(index));
        for (int index = 0; index < count.length(); index++) { assertEquals(1, count.get(index)); }
        var failure = new AssertionError("simulated native allocation failure");
        assertSame(failure, assertThrows(AssertionError.class, () -> CpuRanges.balance(100, index -> {
            if (index == 70) throw failure;
        })));
    }
}
