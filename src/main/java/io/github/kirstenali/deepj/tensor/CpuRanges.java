package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntConsumer;

/** One CPU work item has no parallelism to exploit; avoid a worker handoff and latch. */
public final class CpuRanges {
    private CpuRanges() {}

    public static void forRange(int first, int last, IntConsumer operation) {
        if (last - first == 1) {
            operation.accept(first);
            return;
        }
        var failure = new AtomicReference<Error>();
        DeepJExecutor.forRange(first, last, index -> execute(operation, index, failure));
        if (failure.get() != null) throw failure.get();
    }

    private static void execute(IntConsumer operation, int index, AtomicReference<Error> failure) {
        if (failure.get() != null) return;
        try { operation.accept(index); }
        catch (Error error) { failure.compareAndSet(null, error); }
    }

    static void balance(int count, IntConsumer operation) {
        var cursor = new AtomicInteger();
        int workers = Math.min(count, DeepJExecutor.getNumThreads());
        forRange(0, workers, worker -> consume(cursor, count, operation));
    }

    private static void consume(AtomicInteger cursor, int count, IntConsumer operation) {
        int index;
        while ((index = cursor.getAndIncrement()) < count) { operation.accept(index); }
    }
}
