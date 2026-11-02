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
    public static Enabled enabled() { return new Enabled(); }

    /** Temporarily re-enables graph building, e.g. to recompute a block inside a backward rule. */
    public static final class Enabled implements AutoCloseable {
        private final GradMode previous = CURRENT.get();
        private Enabled() { CURRENT.remove(); }
        @Override public void close() { if (previous != null) CURRENT.set(previous); }
    }

    @Override
    public void close() {
        if (Thread.currentThread() != owner) throw new IllegalStateException("Gradient scope changed threads");
        if (closed) return;
        if (CURRENT.get() != this) throw new IllegalStateException("Gradient scopes must close in reverse order");
        CURRENT.set(previous);
        closed = true;
    }
}
