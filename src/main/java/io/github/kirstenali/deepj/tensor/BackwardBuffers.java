package io.github.kirstenali.deepj.tensor;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;
import java.util.Objects;
import java.util.function.Supplier;

/** Internal lifetime tracking for non-escaping buffers produced by library backward rules. */
public final class BackwardBuffers implements AutoCloseable {
    private static final ThreadLocal<BackwardBuffers> CURRENT = new ThreadLocal<>();
    private final Thread owner = Thread.currentThread();
    private final BackwardBuffers previous = CURRENT.get();
    private final Set<GpuBuffer> buffers = Collections.newSetFromMap(new IdentityHashMap<>());
    private final Set<GpuBuffer> exposed = Collections.newSetFromMap(new IdentityHashMap<>());
    private int depth;
    private boolean closed;

    private BackwardBuffers() { CURRENT.set(this); }

    public static BackwardBuffers begin() { return new BackwardBuffers(); }

    public <T> T capture(Supplier<T> operation) {
        depth++;
        try (var ignored = TensorScope.open()) {
            return operation.get();
        } finally {
            depth--;
        }
    }

    public void expose(Tensor tensor) {
        if (tensor.getGpuTag() instanceof GpuBuffer buffer) exposed.add(buffer);
    }

    static void track(GpuBuffer buffer) {
        BackwardBuffers current = CURRENT.get();
        if (current != null && current.depth > 0) current.buffers.add(buffer);
    }

    /** Releases this private computation's scratch without invalidating unrelated caller tensors. */
    public void releaseTemporaryResources() {
        close();
        buffers.stream().map(buffer -> buffer.owner).filter(Objects::nonNull)
                .distinct().forEach(GraphExecution::flush);
    }

    @Override
    public void close() {
        if (Thread.currentThread() != owner) throw new IllegalStateException("Backward buffers changed threads");
        if (closed) return;
        if (CURRENT.get() != this) throw new IllegalStateException("Backward buffers must close in reverse order");
        for (GpuBuffer buffer : buffers) {
            if (!exposed.contains(buffer) && buffer.owner != null) buffer.owner.retireBackwardBuffer(buffer);
        }
        CURRENT.set(previous);
        closed = true;
    }
}
