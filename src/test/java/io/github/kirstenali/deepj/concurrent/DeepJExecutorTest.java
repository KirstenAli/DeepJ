
package io.github.kirstenali.deepj.concurrent;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

public class DeepJExecutorTest {

    @Test
    void forRange_runsAllIterationsExactlyOnce_sequentialAndParallel() {
        AtomicInteger sum = new AtomicInteger(0);

        // sequential path (parallel disabled)
        DeepJExecutor.setParallelEnabled(false);
        DeepJExecutor.forRange(0, 100, sum::addAndGet);
        Assertions.assertEquals(4950, sum.get());

        // parallel path (parallel enabled)
        sum.set(0);
        DeepJExecutor.setParallelEnabled(true);
        DeepJExecutor.setNumThreads(Math.max(2, DeepJExecutor.getNumThreads()));
        DeepJExecutor.forRange(0, 100, sum::addAndGet);
        Assertions.assertEquals(4950, sum.get());
    }
}
