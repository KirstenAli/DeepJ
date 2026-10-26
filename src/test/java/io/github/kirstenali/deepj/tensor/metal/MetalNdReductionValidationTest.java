package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertThrows;

class MetalNdReductionValidationTest extends MetalDifferentialTestSupport {
    private final List<Tensor> owners = new ArrayList<>();
    private static final long[] LAYOUTS = {2, 6, 6, 0, 2, 3, 3, 1,
            1, 2, 2, 0, 2, 1, 1, 3, 3, 0, 3, 1,
            2, 6, 6, 0, 2, 3, 0, 1, 1, 6, 6, 0, 6, 1};

    @Test
    void reductionsRejectInvalidGroupingOutputWidthsDtypesOperationsAndCorrections() {
        int source = buffer(DType.FLOAT16, 6), output = buffer(DType.FLOAT16, 2);
        int wrongType = buffer(DType.BFLOAT16, 2), wrongSize = buffer(DType.FLOAT16, 3);
        invalid(new int[]{ComputeGraph.OP_LAYOUT_REDUCE, source, output, 0, -1, 0, 0});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_REDUCE, source, output, 0, 3, 0, 0});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_REDUCE, source, output, 0, 1, -1, 0});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_REDUCE, source, output, 0, 1, 2, -1});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_REDUCE, source, output, 0, 1, 0, 1});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_REDUCE, source, output, 0, 1, 5, 0});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_REDUCE, source, wrongType, 0, 1, 0, 0});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_REDUCE, source, wrongSize, 0, 1, 0, 0});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_REDUCE, source, source, 0, 1, 0, 0});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_REDUCE, source, output, 0, 1, 0});
    }

    @Test
    void softmaxRequiresOneAxisIndependentStorageAndACompleteNonoverlappingTypedOutput() {
        int source = buffer(DType.FLOAT16, 6), output = buffer(DType.FLOAT16, 6);
        int wrongType = buffer(DType.FLOAT32, 6);
        invalid(new int[]{ComputeGraph.OP_LAYOUT_SOFTMAX, source, output, 0, 0, 0});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_SOFTMAX, source, output, 0, 0, 3});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_SOFTMAX, source, output, 0, 20, 1});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_SOFTMAX, source, output, 0, 28, 1});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_SOFTMAX, source, wrongType, 0, 0, 1});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_SOFTMAX, source, source, 0, 0, 1});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_SOFTMAX, source, output, 0, 0});
    }

    @Test
    void maximumAdjointsValidateEverySavedValueGradientAndDestinationLayout() {
        int source = buffer(DType.FLOAT16, 6), winner = buffer(DType.FLOAT16, 2);
        int gradient = buffer(DType.FLOAT32, 2), output = buffer(DType.FLOAT32, 6);
        invalidMaximum(source, winner, gradient, output, 0, 3);
        invalidMaximum(source, gradient, gradient, output, 0, 1);
        invalidMaximum(source, winner, winner, output, 0, 1);
        invalidMaximum(source, winner, gradient, output, 20, 1);
        invalidMaximum(source, winner, gradient, source, 0, 1);
        int[] truncated = {ComputeGraph.OP_LAYOUT_MAXIMUM_BACKWARD, source, winner, gradient, output, 0, 8, 8, 0};
        invalid(truncated);
    }

    private static void invalidMaximum(int source, int winner, int gradient, int output, int layout, int retained) {
        invalid(new int[]{ComputeGraph.OP_LAYOUT_MAXIMUM_BACKWARD, source, winner, gradient, output,
                0, 8, 8, layout, retained});
    }

    private int buffer(DType dtype, int count) {
        Tensor owner = new Tensor(dtype, count);
        owners.add(owner);
        int id = metal.graph.ensureStorageBuffer(owner).id;
        metal.graph.flush();
        return id;
    }

    private static void invalid(int[] command) {
        assertThrows(RuntimeException.class,
                () -> MetalNative.nativeFlushLayoutOps(command, command.length, LAYOUTS));
    }
}
