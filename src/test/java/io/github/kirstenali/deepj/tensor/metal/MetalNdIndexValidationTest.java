package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class MetalNdIndexValidationTest extends MetalDifferentialTestSupport {
    private final List<Tensor> owners = new ArrayList<>();
    private static final long[] WORDS = {2, 6, 6, 0, 2, 3, 3, 1,
            2, 6, 6, 0, 2, 3, 3, 1, 3, 2, 2, 0, 2, 2, 0, 0, 2, 3, 0, 2, 1};

    @Test
    void malformedPlansCannotDispatchOutOfBoundsDuplicatedOrReorderedScatterWrites() {
        int target = buffer(DType.FLOAT16, 6), input = buffer(DType.FLOAT16, 6);
        int[] command = {ComputeGraph.OP_LAYOUT_INDEX_ADD, target, input, 0, 8, 16, 1};
        for (int[] corruption : new int[][]{{16, 0}, {17, 0}, {17, 4}, {18, 3}, {21, 0},
                {23, 1}, {24, 4}, {25, 2}, {26, 3}, {27, 0}, {28, 2}}) {
            long[] words = WORDS.clone();
            words[corruption[0]] = corruption[1];
            invalid(command, words);
        }
        long[] overlapping = WORDS.clone();
        overlapping[6] = 0;
        invalid(command, overlapping);
    }

    @Test
    void nativeSelectionRejectsWrongDtypesSizesAliasingAndAxes() {
        int input = buffer(DType.BFLOAT16, 6), output = buffer(DType.BFLOAT16, 6);
        invalid(new int[]{ComputeGraph.OP_LAYOUT_INDEX_SELECT, input, input, 0, 16, 1}, WORDS);
        invalid(new int[]{ComputeGraph.OP_LAYOUT_INDEX_SELECT, input, output, 0, 16, -1}, WORDS);
        invalid(new int[]{ComputeGraph.OP_LAYOUT_INDEX_SELECT, input, output, 0, 16, 2}, WORDS);
        invalid(new int[]{ComputeGraph.OP_LAYOUT_INDEX_SELECT, input, output, 0, 100, 1}, WORDS);
        invalid(new int[]{ComputeGraph.OP_LAYOUT_INDEX_SELECT, input, buffer(DType.FLOAT16, 6), 0, 16, 1}, WORDS);
        invalid(new int[]{ComputeGraph.OP_LAYOUT_INDEX_SELECT, input, buffer(DType.BFLOAT16, 5), 0, 16, 1}, WORDS);
        invalid(new int[]{ComputeGraph.OP_LAYOUT_INDEX_SELECT, input, output, 0, 16}, WORDS);
    }

    @Test
    void nativeUpdatesRejectAliasingNarrowingShapeAndTruncatedMetadata() {
        int target = buffer(DType.FLOAT16, 6), input = buffer(DType.FLOAT16, 6);
        invalid(new int[]{ComputeGraph.OP_LAYOUT_INDEX_ADD, target, target, 0, 8, 16, 1}, WORDS);
        invalid(new int[]{ComputeGraph.OP_LAYOUT_INDEX_ADD, target, buffer(DType.FLOAT32, 6), 0, 8, 16, 1}, WORDS);
        invalid(new int[]{ComputeGraph.OP_LAYOUT_INDEX_ADD, target, input, 0, 8, 16, 0}, WORDS);
        invalid(new int[]{ComputeGraph.OP_LAYOUT_INDEX_ADD, target, input, 0, 8, 16, 1}, new long[2]);
        invalid(new int[]{ComputeGraph.OP_LAYOUT_INDEX_ADD, target, input, 0, 8, 16}, WORDS);
    }

    @Test
    void invalidPublicMutationsPreserveVersionsAndOriginalStorage() {
        Tensor target = new Tensor(DType.FLOAT16, 2, 3);
        assertThrows(IndexOutOfBoundsException.class, () -> target.indexSelect(1, -1));
        assertThrows(IndexOutOfBoundsException.class, () -> target.indexSelect(1, 3));
        assertThrows(IllegalArgumentException.class,
                () -> target.indexAddInPlace(1, new int[]{1}, new Tensor(2, 1)));
        assertEquals(0, target.gradientState().version());
        assertNull(target.getGpuTag());
        Tensor overlapping = Tensor.from(new float[]{1}).expand(3);
        assertThrows(IllegalStateException.class,
                () -> overlapping.indexAddInPlace(0, new int[]{1}, new Tensor(1)));
    }

    private int buffer(DType dtype, int count) {
        Tensor owner = new Tensor(dtype, count);
        owners.add(owner);
        int id = metal.graph.ensureStorageBuffer(owner).id;
        metal.graph.flush();
        return id;
    }

    private static void invalid(int[] command, long[] words) {
        assertThrows(RuntimeException.class,
                () -> MetalNative.nativeFlushLayoutOps(command, command.length, words));
    }
}
