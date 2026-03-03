package io.github.kirstenali.deepj.concurrent;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.IntConsumer;

public final class DeepJExecutor {

    private static volatile int parallelThreshold = 64;
    private static final AtomicInteger tid = new AtomicInteger(1);

    private static ThreadFactory daemonFactory() {
        return r -> {
            Thread t = new Thread(r, "deepj-" + tid.getAndIncrement());
            t.setDaemon(true);
            return t;
        };
    }

    private static int defaultThreads() {
        return Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
    }

    private static ThreadPoolExecutor newExecutor(int nThreads) {
        ThreadPoolExecutor p = new ThreadPoolExecutor(
                nThreads, nThreads,
                30L, TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(),
                daemonFactory()
        );

        p.allowCoreThreadTimeOut(true);
        return p;
    }

    private static volatile ThreadPoolExecutor exec = newExecutor(defaultThreads());

    private DeepJExecutor() {}

    public static synchronized void setNumThreads(int n) {
        if (n < 1) throw new IllegalArgumentException("n must be >= 1");
        ThreadPoolExecutor old = exec;
        exec = newExecutor(n);
        old.shutdown();
    }

    /** Current number of threads used by DeepJ. */
    public static int getNumThreads() {
        return exec.getCorePoolSize();
    }

    public static void setParallelThreshold(int n) {
        if (n < 0) throw new IllegalArgumentException("threshold must be >= 0");
        parallelThreshold = n;
    }

    /** Current minimum iteration count before DeepJ uses parallel execution. */
    public static int getParallelThreshold() {
        return parallelThreshold;
    }

    /**
     * Clean shutdown (good for CLI apps/tests). Daemon threads already let the JVM exit,
     * but shutdown avoids work continuing after main finishes.
     */
    public static void shutdown() {
        ThreadPoolExecutor pool = exec;
        pool.shutdown();
        try {
            if (!pool.awaitTermination(2, TimeUnit.SECONDS)) {
                pool.shutdownNow();
            }
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            pool.shutdownNow();
        }
    }

    public static void forRange(int startInclusive, int endExclusive, IntConsumer body) {
        int n = endExclusive - startInclusive;
        if (n <= 0) return;

        int threads = exec.getCorePoolSize();
        if (threads <= 1 || n < parallelThreshold) {
            for (int i = startInclusive; i < endExclusive; i++) body.accept(i);
            return;
        }

        runParallelChunked(startInclusive, endExclusive, threads, body);
    }

    private static void runParallelChunked(int startInclusive, int endExclusive, int threads, IntConsumer body) {
        int n = endExclusive - startInclusive;

        int chunks = Math.min(threads, n);
        int chunkSize = (n + chunks - 1) / chunks;

        CountDownLatch latch = new CountDownLatch(chunks);

        // fail-fast: once any chunk throws, other chunks can stop early
        AtomicBoolean cancelled = new AtomicBoolean(false);

        AtomicReference<RuntimeException> firstError = new AtomicReference<>();

        submitChunkedRange(startInclusive, endExclusive, chunks, chunkSize, latch, cancelled, firstError, body);

        awaitLatch(latch, cancelled);

        RuntimeException ex = firstError.get();
        if (ex != null) throw ex;
    }

    /**
     * Submits chunk tasks without storing chunk ranges (O(1) extra memory).
     */
    private static void submitChunkedRange(
            int startInclusive,
            int endExclusive,
            int chunks,
            int chunkSize,
            CountDownLatch latch,
            AtomicBoolean cancelled,
            AtomicReference<RuntimeException> firstError,
            IntConsumer body
    ) {
        for (int t = 0; t < chunks; t++) {
            int s = startInclusive + t * chunkSize;
            int e = Math.min(endExclusive, s + chunkSize);

            if (s >= e) {
                latch.countDown();
                continue;
            }

            exec.execute(() -> {
                try {
                    for (int i = s; i < e && !cancelled.get(); i++) {
                        body.accept(i);
                    }
                } catch (RuntimeException ex) {
                    cancelled.set(true);
                    firstError.compareAndSet(null, ex);
                } finally {
                    latch.countDown();
                }
            });
        }
    }

    private static void awaitLatch(CountDownLatch latch, AtomicBoolean cancelled) {
        try {
            latch.await();
        } catch (InterruptedException ie) {
            cancelled.set(true);
            Thread.currentThread().interrupt();
            throw new RuntimeException(ie);
        }
    }

    public static Range range(int startInclusive, int endExclusive) {
        return new Range(startInclusive, endExclusive);
    }

    public static final class Range {
        private final int startInclusive;
        private final int endExclusive;
        private boolean parallel;

        private Range(int startInclusive, int endExclusive) {
            this.startInclusive = startInclusive;
            this.endExclusive = endExclusive;
        }

        public Range parallel() {
            this.parallel = true;
            return this;
        }

        public Range sequential() {
            this.parallel = false;
            return this;
        }

        public void forEach(IntConsumer body) {
            if (!parallel) {
                for (int i = startInclusive; i < endExclusive; i++) body.accept(i);
            } else {
                DeepJExecutor.forRange(startInclusive, endExclusive, body);
            }
        }
    }
}