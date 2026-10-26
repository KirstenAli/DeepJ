package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class GpuCrossEntropyRecordingTest {
    private final ComputeGraph graph = new ComputeGraph(new GpuStorageRuntime());

    @Test
    void fusedLossAndGradientHaveFloat32LazyStorageAndExactMetadata() {
        Tensor logits = new Tensor(DType.FLOAT16, 2, 1, 3);
        int[] targets = {2, 0};
        boolean[] mask = {false, true};
        var result = graph.layoutCrossEntropy(logits, targets, mask, 3);
        targets[1] = 99;
        mask[0] = true;
        assertEquals(ComputeGraph.OP_LAYOUT_CROSS_ENTROPY, graph.cmdStream[0]);
        assertEquals(12, graph.cmdPos);
        assertEquals(3, graph.pendingAllocs.size());
        assertEquals(DType.FLOAT32, result.gradient().dtype());
        assertArrayEquals(logits.shape(), result.gradient().shape());
        assertFalse(result.gradient().memory().hasHostStorage());
        assertEquals(ComputeGraph.OP_MULTIPLY_SCALAR, graph.cmdStream[7]);
        assertEquals(graph.cmdStream[8], graph.cmdStream[9]);
        assertArrayEquals(new long[]{2, 1, 1, 2, 0, 0, 1},
                java.util.Arrays.copyOfRange(graph.layouts.snapshot(), 10, 17));
    }

    @Test
    void lossOnlyDoesNotAllocateAFullGradientAndGradientOnlyDoesNotAllocateRowLosses() {
        Tensor logits = new Tensor(DType.BFLOAT16, 2, 1, 1000);
        var losses = graph.layoutCrossEntropy(logits, new int[]{1, 2}, null, 1);
        assertNull(losses.gradient());
        assertEquals(2, graph.pendingAllocs.size());
        assertEquals(8, losses.loss().storageBytes());
        graph.layoutCrossEntropy(logits, new int[]{1, 2}, null, 2);
        assertEquals(3, graph.pendingAllocs.size());
        assertEquals(graph.cmdStream[9], graph.cmdStream[10]);
    }

    @Test
    void largeIntegerTargetsAreNeverConvertedToFloat() {
        long[] plan = CrossEntropyPlan.describe(new Tensor(DType.FLOAT16, 16_777_218),
                new int[]{16_777_217}, null);
        assertArrayEquals(new long[]{1, 1, 0, 16_777_217, 1}, plan);
        GraphLayouts layouts = new GraphLayouts();
        layouts.addWords(plan);
        plan[3] = 0;
        assertEquals(16_777_217, layouts.snapshot()[3]);
    }

    @Test
    void invalidGeometryModeTargetsAndMasksFailBeforeBufferAllocation() {
        Tensor logits = new Tensor(DType.FLOAT16, 2, 1, 3);
        assertThrows(IllegalArgumentException.class,
                () -> graph.layoutCrossEntropy(logits, new int[]{0, 1}, null, 0));
        assertThrows(IllegalArgumentException.class,
                () -> graph.layoutCrossEntropy(logits, new int[]{0, 3}, new boolean[]{true, false}, 3));
        assertThrows(IllegalArgumentException.class,
                () -> graph.layoutCrossEntropy(logits, new int[]{0, 1}, new boolean[2], 3));
        assertThrows(IllegalArgumentException.class,
                () -> graph.layoutCrossEntropy(new Tensor(DType.FLOAT16, 0, 3), new int[0], null, 1));
        assertTrue(graph.pendingAllocs.isEmpty());
    }
}
