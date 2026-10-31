package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertThrows;

class MetalNdMatmulValidationTest extends MetalDifferentialTestSupport {
    private final List<Tensor> owners = new ArrayList<>();
    private static final long[] LAYOUTS = {2, 6, 6, 0, 2, 3, 3, 1,
            2, 6, 6, 0, 3, 2, 2, 1, 1, 6, 6, 0, 6, 1,
            3, 6, 6, 0, 1, 3, 2, 6, 2, 1};

    @Test
    void nativeMatmulRejectsInvalidRanksBatchShapesInnerWidthsAndDescriptors() {
        int left = buffer(DType.FLOAT16, 6), right = buffer(DType.FLOAT16, 6);
        int output = buffer(DType.FLOAT16, 4);
        invalid(left, right, output, 0, 0);
        invalid(left, right, output, 16, 8);
        invalid(left, right, output, 0, 22);
        invalid(left, right, output, -1, 8);
        invalid(left, right, output, 0, 100);
        assertThrows(RuntimeException.class, () -> MetalNative.nativeFlushLayoutOps(
                new int[]{ComputeGraph.OP_LAYOUT_MATMUL, left, right, output, 0}, 5, LAYOUTS));
    }

    @Test
    void outputMustBeIndependentCorrectlySizedAndHaveThePromotedDtype() {
        int left = buffer(DType.FLOAT16, 6), right = buffer(DType.FLOAT16, 6);
        invalid(left, right, buffer(DType.FLOAT16, 3), 0, 8);
        invalid(left, right, buffer(DType.BFLOAT16, 4), 0, 8);
        invalid(left, right, buffer(DType.FLOAT32, 4), 0, 8);
        invalid(left, right, left, 0, 8);
        int mixed = buffer(DType.BFLOAT16, 6);
        invalid(left, mixed, buffer(DType.FLOAT16, 4), 0, 8);
        invalid(left, mixed, right, 0, 8);
    }

    @Test
    void publicGeometryValidationDoesNotUploadOrAllocateInvalidInputs() {
        assertThrows(IllegalArgumentException.class, () -> metal.matmul(new Tensor(), new Tensor(1)));
        assertThrows(IllegalArgumentException.class, () -> metal.matmul(new Tensor(3), new Tensor(4)));
        assertThrows(IllegalArgumentException.class,
                () -> metal.matmul(new Tensor(2, 2, 3), new Tensor(3, 3, 4)));
        assertThrows(IllegalArgumentException.class,
                () -> metal.batchedMatmul(new Tensor(DType.FLOAT16, 4, 2), new Tensor(4, 2), 3, false, false));
    }

    private int buffer(DType dtype, int count) {
        Tensor owner = new Tensor(dtype, count);
        owners.add(owner);
        int id = metal.graph.ensureStorageBuffer(owner).id;
        metal.graph.flush();
        return id;
    }

    private static void invalid(int a, int b, int output, int first, int second) {
        int[] command = {ComputeGraph.OP_LAYOUT_MATMUL, a, b, output, first, second};
        assertThrows(RuntimeException.class,
                () -> MetalNative.nativeFlushLayoutOps(command, command.length, LAYOUTS));
    }
}
