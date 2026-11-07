package io.github.kirstenali.deepj.layers.transformer.qwen3;

import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

/** Shares position tables across layers for one forward call, without retaining device memory. */
public final class Qwen3RotaryTables implements AutoCloseable {
    private static final ThreadLocal<Qwen3RotaryTables> CURRENT = new ThreadLocal<>();
    private final Qwen3RotaryTables previous = CURRENT.get();
    private final Map<Key, Tensor[]> tables = new HashMap<>();

    private Qwen3RotaryTables() { CURRENT.set(this); }
    public static Qwen3RotaryTables open() { return new Qwen3RotaryTables(); }

    static Tensor[] get(int width, double theta, int length, int offset, Supplier<Tensor[]> build) {
        return get("qwen3", width, theta, length, offset, build);
    }

    /** Distinguishes model-specific table shapes and rounding conventions within the shared forward scope. */
    public static Tensor[] get(String convention, int width, double theta, int length, int offset,
                               Supplier<Tensor[]> build) {
        Qwen3RotaryTables scope = CURRENT.get();
        if (scope == null) return build.get();
        return scope.tables.computeIfAbsent(new Key(convention, width, theta, length, offset), ignored -> build.get());
    }

    @Override public void close() {
        if (CURRENT.get() != this) throw new IllegalStateException("Rotary scopes must close in order");
        tables.clear();
        if (previous == null) CURRENT.remove();
        else CURRENT.set(previous);
    }

    private record Key(String convention, int width, double theta, int length, int offset) {}
}
