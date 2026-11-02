package io.github.kirstenali.deepj.distributed;

import java.util.List;
import java.util.Objects;

/** One-worker implementation for validation and the same opt-in training entry point. */
public final class LocalProcessGroup implements ProcessGroup {
    private boolean closed;
    @Override public int rank() { return 0; }
    @Override public int worldSize() { return 1; }
    @Override public void sum(String key, float[] values) {
        check(key);
        Objects.requireNonNull(values);
    }
    @Override public void broadcast(String key, float[] values) { sum(key, values); }
    @Override public List<byte[]> gather(String key, byte[] value) {
        check(key);
        return List.of(value.clone());
    }
    @Override public void abort(String reason) { closed = true; }
    @Override public void close() { closed = true; }

    private void check(String key) {
        Objects.requireNonNull(key, "key");
        if (closed) throw new CollectiveException("Process group is closed");
    }
}
