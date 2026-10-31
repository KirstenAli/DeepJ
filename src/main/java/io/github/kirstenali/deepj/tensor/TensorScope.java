package io.github.kirstenali.deepj.tensor;

import java.util.function.Supplier;

/** Bounds internal view lifetimes, without releasing saved or explicitly retained storage.
 * Views created inside a scope are temporary; return them through {@link #escape(Tensor)}.
 * Closing a scope permits later device cleanup, but never performs a download or allocation.
 */
public final class TensorScope implements AutoCloseable {
    private static final ThreadLocal<TensorScope> CURRENT = new ThreadLocal<>();
    private final Thread owner = Thread.currentThread();
    private final TensorScope previous = CURRENT.get();
    private boolean closed;

    private TensorScope() { CURRENT.set(this); }

    public static TensorScope open() { return new TensorScope(); }

    static TensorScope current() { return CURRENT.get(); }

    boolean active() { return !closed; }

    public Tensor escape(Tensor tensor) {
        requireCurrent();
        tensor.memory().escapeAlias(tensor.gradientState(), this, previous);
        return tensor;
    }

    /** User callbacks must not inherit the library's private lifetime boundary. */
    public static <T> T external(Supplier<T> operation) {
        TensorScope previous = CURRENT.get();
        CURRENT.remove();
        try { return operation.get(); }
        finally { restore(previous); }
    }

    private static void restore(TensorScope scope) {
        if (scope == null) CURRENT.remove();
        else CURRENT.set(scope);
    }

    private void requireCurrent() {
        if (Thread.currentThread() != owner || CURRENT.get() != this || closed) {
            throw new IllegalStateException("Tensor scopes must be used in order on their owning thread");
        }
    }

    @Override public void close() {
        if (closed) return;
        requireCurrent();
        closed = true;
        restore(previous);
    }
}
