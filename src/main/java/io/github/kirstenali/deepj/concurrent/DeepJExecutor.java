package io.github.kirstenali.deepj.concurrent;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.IntConsumer;

public final class DeepJExecutor {

    private static volatile boolean parallelEnabled = true;
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

    public static int getNumThreads() {
        return exec.getCorePoolSize();
    }

    public static void setParallelEnabled(boolean enabled) {
        parallelEnabled = enabled;
    }

    public static boolean isParallelEnabled() {
        return parallelEnabled;
    }

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
        if (threads <= 1 || !parallelEnabled) {
            for (int i = startInclusive; i < endExclusive; i++) {
                body.accept(i);
            }
            return;
        }

        runParallelChunked(startInclusive, endExclusive, threads, body);
    }

    private static void runParallelChunked(int startInclusive, int endExclusive, int threads, IntConsumer body) {
        int n = endExclusive - startInclusive;

        int chunks = Math.min(threads, n);
        int chunkSize = (n + chunks - 1) / chunks;

        CountDownLatch latch = new CountDownLatch(chunks);

        AtomicBoolean cancelled = new AtomicBoolean(false);

        AtomicReference<RuntimeException> firstError = new AtomicReference<>();

        submitChunkedRange(startInclusive, endExclusive, chunks, chunkSize, latch, cancelled, firstError, body);

        awaitLatch(latch, cancelled);

        RuntimeException ex = firstError.get();
        if (ex != null) throw ex;
    }

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
            exec.execute(() -> runChunk(s, e, latch, cancelled, firstError, body));
        }
    }

    private static void runChunk(int start, int end, CountDownLatch latch, AtomicBoolean cancelled,
                                 AtomicReference<RuntimeException> firstError, IntConsumer body) {
        try {
            for (int i = start; i < end && !cancelled.get(); i++) {
                body.accept(i);
            }
        } catch (RuntimeException ex) {
            cancelled.set(true);
            firstError.compareAndSet(null, ex);
        } finally {
            latch.countDown();
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
}
