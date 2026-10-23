package io.github.kirstenali.deepj.autograd;

/** Thread-local, nestable scope for computations that must not build a backward graph. */
public final class GradMode implements AutoCloseable {
    private static final ThreadLocal<GradMode> CURRENT = new ThreadLocal<>();
    private final Thread owner = Thread.currentThread();
    private final GradMode previous;
    private boolean closed;

    private GradMode() {
        previous = CURRENT.get();
        CURRENT.set(this);
    }

    public static boolean isEnabled() { return CURRENT.get() == null; }
    public static GradMode noGrad() { return new GradMode(); }

    @Override
    public void close() {
        if (Thread.currentThread() != owner) throw new IllegalStateException("Gradient scope changed threads");
        if (closed) return;
        if (CURRENT.get() != this) throw new IllegalStateException("Gradient scopes must close in reverse order");
        CURRENT.set(previous);
        closed = true;
    }
}
