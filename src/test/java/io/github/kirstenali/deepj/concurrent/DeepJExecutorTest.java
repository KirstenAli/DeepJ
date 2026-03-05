
package io.github.kirstenali.deepj.concurrent;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

public class DeepJExecutorTest {

    @Test
    void forRange_runsAllIterationsExactlyOnce_sequentialAndParallel() {
        AtomicInteger sum = new AtomicInteger(0);

        // sequential path
        DeepJExecutor.setParallelThreshold(Integer.MAX_VALUE);
        DeepJExecutor.forRange(0, 100, sum::addAndGet);
        Assertions.assertEquals(4950, sum.get());

        // parallel path (force)
        sum.set(0);
        DeepJExecutor.setParallelThreshold(1);
        DeepJExecutor.setNumThreads(Math.max(2, DeepJExecutor.getNumThreads()));
        DeepJExecutor.forRange(0, 100, sum::addAndGet);
        Assertions.assertEquals(4950, sum.get());
    }
}
