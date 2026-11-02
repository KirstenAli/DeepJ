package io.github.kirstenali.deepj.data;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/** Builds upcoming batches on a daemon thread so host batching overlaps device work. */
public final class PrefetchingBatchSource implements BatchSource, AutoCloseable {
    private final BatchSource source;
    private final int batchSize;
    private final BlockingQueue<Batch> ready;
    private final AtomicReference<Throwable> failure = new AtomicReference<>();
    private final Thread worker;
    private volatile boolean closed;

    public PrefetchingBatchSource(BatchSource source, int batchSize, int depth) {
        if (source == null) throw new IllegalArgumentException("source must not be null");
        if (batchSize <= 0 || depth <= 0) throw new IllegalArgumentException("batchSize and depth must be > 0");
        this.source = source;
        this.batchSize = batchSize;
        this.ready = new ArrayBlockingQueue<>(depth);
        this.worker = start();
    }

    @Override
    public Batch nextBatch(int requested) {
        if (requested != batchSize) throw new IllegalArgumentException("Prefetch batch size is fixed at " + batchSize);
        if (closed) throw new IllegalStateException("Prefetch source is closed");
        return take();
    }

    @Override
    public void close() {
        closed = true;
        worker.interrupt();
        ready.clear();
    }

    private Thread start() {
        Thread thread = new Thread(this::produce, "deepj-prefetch");
        thread.setDaemon(true);
        thread.start();
        return thread;
    }

    private void produce() {
        try {
            while (!closed) { ready.put(source.nextBatch(batchSize)); }
        } catch (InterruptedException ignored) {
            Thread.currentThread().interrupt();
        } catch (Throwable error) {
            failure.set(error);
        }
    }

    private Batch take() {
        try {
            Batch batch;
            while ((batch = ready.poll(10, TimeUnit.MILLISECONDS)) == null) { rethrowFailure(); }
            return batch;
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while waiting for a batch", interrupted);
        }
    }

    private void rethrowFailure() {
        Throwable error = failure.get();
        if (error == null) return;
        if (error instanceof RuntimeException runtime) throw runtime;
        throw new IllegalStateException("Batch prefetch failed", error);
    }
}

