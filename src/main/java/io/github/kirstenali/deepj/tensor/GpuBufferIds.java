package io.github.kirstenali.deepj.tensor;

import java.util.Arrays;

/** IDs are unique while leased, then recycled only after their native buffers have been released. */
final class GpuBufferIds {
    static final GpuBufferIds SHARED = new GpuBufferIds(0);
    private long next;
    private int[] available = new int[32];
    private int count;

    GpuBufferIds(long next) { this.next = next; }

    synchronized int acquire() {
        if (count > 0) return available[--count];
        if (next > Integer.MAX_VALUE) throw new IllegalStateException("GPU buffer id space exhausted");
        return (int) next++;
    }

    synchronized void release(int[] ids, int length) {
        if (length == 0) return;
        if (count + length > available.length) {
            available = Arrays.copyOf(available, Math.max(available.length * 2, count + length));
        }
        System.arraycopy(ids, 0, available, count, length);
        count += length;
    }
}
