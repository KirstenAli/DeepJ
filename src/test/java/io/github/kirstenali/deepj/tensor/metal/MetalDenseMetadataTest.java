package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Random;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MetalDenseMetadataTest {
    @BeforeEach
    void cpuMetadataOnly() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @EnumSource(DType.class)
    void cachedCoverageAndMetalRoutingMatchIndependentAddresses(DType dtype) {
        var random = new Random(2026);
        for (int rank = 0; rank <= 6; rank++) {
            for (int sample = 0; sample < 4; sample++) {
                int[] shape = random.ints(rank, 1, 4).toArray();
                checkViews(new Tensor(dtype, shape));
                if (rank > 0) {
                    shape[rank - 1] = 0;
                    checkViews(new Tensor(dtype, shape));
                }
            }
        }
    }

    private static void checkViews(Tensor base) {
        check(base);
        check(base.unsqueeze(0));
        if (base.rank() == 0) return;
        int[] axes = IntStream.range(0, base.rank()).map(axis -> base.rank() - axis - 1).toArray();
        check(base.permute(axes));
        check(base.slice(0, 0, base.size(0), 2));
        int[] expanded = new int[base.rank() + 1];
        expanded[0] = 2;
        System.arraycopy(base.shape(), 0, expanded, 1, base.rank());
        check(base.unsqueeze(0).expand(expanded));
        if (base.size(0) > 0) check(base.select(0, 0));
    }

    private static void check(Tensor tensor) {
        boolean full = tensor.storageOffset() == 0 && consecutiveAddresses(tensor)
                && tensor.storageBytes() == tensor.dtype().storageBytes(tensor.numel());
        assertEquals(full, tensor.coversStorage());
        boolean dense = full && tensor.numel() > 0 && tensor.dtype() == DType.FLOAT32;
        assertEquals(dense, MetalDenseOps.dense(tensor));
        assertEquals(dense && tensor.rank() == 2, MetalDenseOps.matrix(tensor));
    }

    private static boolean consecutiveAddresses(Tensor tensor) {
        int[] dimensions = tensor.shape();
        long[] strides = tensor.strides();
        for (int index = 0; index < tensor.numel(); index++) {
            int remainder = index;
            long address = tensor.storageOffset();
            for (int axis = dimensions.length - 1; axis >= 0; axis--) {
                address += (remainder % dimensions[axis]) * strides[axis];
                remainder /= dimensions[axis];
            }
            if (address != tensor.storageOffset() + index) return false;
        }
        return true;
    }
}
