package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;

import java.util.HashMap;
import java.util.Map;

/** Optional process-wide guard for managed native buffers, not a total process-memory limit. */
final class MetalAllocationBudget {
    private final long limit;
    private final Map<Integer, Long> allocations = new HashMap<>();
    private long live;
    private long peak;
    private boolean poisoned;

    MetalAllocationBudget(long limit) {
        if (limit < 0) throw new IllegalArgumentException("Metal buffer budget must not be negative");
        this.limit = limit;
    }

    static MetalAllocationBudget configured() {
        return new MetalAllocationBudget(Long.parseLong(System.getProperty("deepj.metal.maxBufferBytes", "0")));
    }

    boolean enabled() { return limit != 0; }

    synchronized void allocate(int[] ids, int[] sizes, DType[] types, int count,
                                Runnable operation, Runnable rollback) {
        if (limit == 0) {
            operation.run();
            return;
        }
        var pending = MetalAllocationRequests.capture(ids, sizes, types, count);
        pending.requireUnallocated(allocations);
        long requested = pending.totalBytes();
        requireCapacity(requested);
        attempt(operation, rollback);
        pending.commitTo(allocations);
        live += requested;
        peak = Math.max(peak, live);
    }

    private void requireCapacity(long requested) {
        if (poisoned) throw new IllegalStateException("Metal budget requires a fresh process after failed cleanup");
        if (requested > limit - live) {
            throw new IllegalStateException("Metal allocation budget exceeded: live=" + live
                    + " requested=" + requested + " limit=" + limit);
        }
    }

    private void attempt(Runnable operation, Runnable rollback) {
        try { operation.run(); }
        catch (RuntimeException | Error failure) {
            try { rollback.run(); }
            catch (RuntimeException | Error cleanup) {
                poisoned = true;
                failure.addSuppressed(cleanup);
            }
            throw failure;
        }
    }

    synchronized void release(int[] ids, int count, Runnable operation) {
        operation.run();
        if (limit == 0) return;
        for (int index = 0; index < count; index++) {
            Long bytes = allocations.remove(ids[index]);
            if (bytes != null) live -= bytes;
        }
    }

    synchronized long liveBytes() { return live; }
    synchronized long peakBytes() { return peak; }
}
