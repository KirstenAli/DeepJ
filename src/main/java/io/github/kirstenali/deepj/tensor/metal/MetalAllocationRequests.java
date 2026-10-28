package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;

import java.util.Arrays;
import java.util.Map;

/** Captures a batch before native allocation without a temporary boxed allocation ledger. */
final class MetalAllocationRequests {
    private final int[] ids;
    private final long[] bytes;
    private final long totalBytes;

    private MetalAllocationRequests(int[] ids, long[] bytes, long totalBytes) {
        this.ids = ids;
        this.bytes = bytes;
        this.totalBytes = totalBytes;
    }

    static MetalAllocationRequests capture(int[] ids, int[] sizes, DType[] types, int count) {
        int[] capturedIds = Arrays.copyOf(ids, count);
        long[] bytes = new long[count];
        long total = 0;
        for (int index = 0; index < count; index++) {
            int width = types == null ? Float.BYTES : types[index].bytes();
            bytes[index] = (long) sizes[index] * width;
            total = Math.addExact(total, bytes[index]);
        }
        return new MetalAllocationRequests(capturedIds, bytes, total);
    }

    void requireUnallocated(Map<Integer, Long> allocations) {
        int[] ordered = ids.clone();
        Arrays.sort(ordered);
        for (int index = 0; index < ordered.length; index++) {
            int id = ordered[index];
            if (allocations.containsKey(id) || (index > 0 && id == ordered[index - 1])) {
                throw new IllegalArgumentException("Duplicate managed Metal buffer: " + id);
            }
        }
    }

    void commitTo(Map<Integer, Long> allocations) {
        for (int index = 0; index < ids.length; index++) {
            allocations.put(ids[index], bytes[index]);
        }
    }

    long totalBytes() { return totalBytes; }
}
