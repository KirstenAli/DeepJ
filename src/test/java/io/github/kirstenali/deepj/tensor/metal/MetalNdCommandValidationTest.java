package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertThrows;

class MetalNdCommandValidationTest extends MetalDifferentialTestSupport {
    private final List<Tensor> owners = new ArrayList<>();
    private static final long[] LAYOUTS = {2, 4, 4, 0, 2, 2, 2, 1,
            2, 4, 4, 0, 2, 2, 0, 1, 1, 4, 4, 0, 4, 1};

    @Test
    void binaryCommandsRejectInvalidTagsDtypesShapesOutputsAndTruncation() {
        int left = buffer(DType.FLOAT16), right = buffer(DType.BFLOAT16);
        int output = buffer(DType.FLOAT32), compact = buffer(DType.FLOAT16);
        invalid(new int[]{ComputeGraph.OP_LAYOUT_BINARY, left, right, output, 0, 0, -1});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_BINARY, left, right, output, 0, 0, ComputeGraph.OP_EXP});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_BINARY, left, right, compact, 0, 0, ComputeGraph.OP_ADD});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_BINARY, left, right, left, 0, 0, ComputeGraph.OP_ADD});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_BINARY, left, right, output, 0, 16, ComputeGraph.OP_ADD});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_BINARY, left, right, output, -1, 0, ComputeGraph.OP_ADD});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_BINARY, left, right, output, 0, 0});
    }

    @Test
    void unaryCommandsRetainDtypeAndRejectInvalidTagsAndAliasedOutputs() {
        int source = buffer(DType.FLOAT16), output = buffer(DType.FLOAT16), wrongType = buffer(DType.FLOAT32);
        invalid(new int[]{ComputeGraph.OP_LAYOUT_UNARY, source, output, 0, -1, 0, 0});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_UNARY, source, output, 0, ComputeGraph.OP_MATMUL, 0, 0});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_UNARY, source, source, 0, ComputeGraph.OP_NEG, 0, 0});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_UNARY, source, wrongType, 0, ComputeGraph.OP_NEG, 0, 0});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_UNARY, source, output, 0, ComputeGraph.OP_NEG, 0});
    }

    @Test
    void assignmentsRejectOverlappingDestinationsShapeMismatchAndMissingSnapshots() {
        int source = buffer(DType.FLOAT16), target = buffer(DType.FLOAT16);
        invalid(new int[]{ComputeGraph.OP_LAYOUT_ASSIGN, source, target, 0, 8});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_ASSIGN, source, target, 0, 16});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_ASSIGN, source, source, 0, 0});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_ASSIGN, source, target, 0, -1});
        invalid(new int[]{ComputeGraph.OP_LAYOUT_ASSIGN, source, target, 0});
    }

    private int buffer(DType dtype) {
        Tensor owner = new Tensor(dtype, 2, 2);
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
