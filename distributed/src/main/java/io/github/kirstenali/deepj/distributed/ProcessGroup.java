package io.github.kirstenali.deepj.distributed;

import java.util.List;

/** Ordered, blocking collectives. Every rank must call the same keys in the same order. */
public interface ProcessGroup extends AutoCloseable {
    int rank();
    int worldSize();
    void sum(String key, float[] values);
    void broadcast(String key, float[] values);
    List<byte[]> gather(String key, byte[] value);
    void abort(String reason);
    @Override void close();

    default void barrier(String key) { gather(key, new byte[0]); }
}
