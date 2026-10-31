package io.github.kirstenali.deepj.tensor;

import java.util.List;
import java.util.Objects;

/** Validates physical allocations without flattening logical tensor dimensions. */
public final class GpuAllocations {
    private GpuAllocations() {}

    public static void requireValid(int[] ids, int[] sizes, DType[] dtypes, int count) {
        Objects.requireNonNull(ids);
        Objects.requireNonNull(sizes);
        Objects.requireNonNull(dtypes);
        if (count < 0 || count > ids.length || count > sizes.length || count > dtypes.length) {
            throw new IllegalArgumentException("Invalid allocation count");
        }
        for (int index = 0; index < count; index++) {
            if (ids[index] < 0 || sizes[index] <= 0) throw new IllegalArgumentException("Invalid GPU allocation");
            Objects.requireNonNull(dtypes[index], "dtype");
        }
    }

    static boolean allFloat32(DType[] dtypes, int count) {
        for (int index = 0; index < count; index++) {
            if (dtypes[index] != DType.FLOAT32) return false;
        }
        return true;
    }

    static DType type(int[] allocation) {
        return allocation.length == 2 ? DType.FLOAT32 : DType.fromStorageCode(allocation[2]);
    }

    static void allocate(GpuRuntime runtime, List<int[]> allocations) {
        int count = allocations.size();
        int[] ids = new int[count], sizes = new int[count];
        DType[] types = new DType[count];
        for (int index = 0; index < count; index++) {
            int[] allocation = allocations.get(index);
            ids[index] = allocation[0];
            sizes[index] = allocation[1];
            types[index] = type(allocation);
        }
        if (allFloat32(types, count)) runtime.allocBuffers(ids, sizes, count);
        else runtime.allocTypedBuffers(ids, sizes, types, count);
    }
}
