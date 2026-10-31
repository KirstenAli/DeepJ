package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class GpuTrainingRecordingTest {
    private final ComputeGraph graph = new ComputeGraph(new GpuStorageRuntime());

    @Test
    void normalizationKeepsTypedOutputsAndFloat32SavedStatistics() {
        Tensor input = new Tensor(DType.FLOAT16, 2, 3, 4), gamma = new Tensor(DType.FLOAT16, 4);
        var result = graph.layoutNormalize(input, gamma, null, 1e-5f, false);
        assertArrayEquals(input.shape(), result.output().shape());
        assertArrayEquals(new int[]{2, 3, 1}, result.denominator().shape());
        assertEquals(DType.FLOAT16, result.output().dtype());
        assertEquals(DType.FLOAT32, result.normalized().dtype());
        assertEquals(DType.FLOAT32, result.denominator().dtype());
        assertEquals(ComputeGraph.OP_LAYOUT_NORMALIZE, graph.cmdStream[0]);
        assertEquals(5, graph.pendingAllocs.size());
        assertFalse(result.normalized().memory().hasHostStorage());
        Tensor gradient = graph.layoutNormalizationBackward(new Tensor(input.shape()), result.normalized(),
                result.denominator(), gamma, false);
        assertEquals(DType.FLOAT32, gradient.dtype());
        assertEquals(ComputeGraph.OP_LAYOUT_NORMALIZATION_BACKWARD, graph.cmdStream[12]);
    }

    @Test
    void swigluAdjointsKeepTheThreeInputLayoutsAndTwoFloat32Outputs() {
        Tensor gate = new Tensor(DType.FLOAT16, 2, 1, 3), up = new Tensor(DType.BFLOAT16, 2, 1, 3);
        var result = graph.layoutSwiGluBackward(new Tensor(2, 1, 3), gate, up);
        assertEquals(ComputeGraph.OP_LAYOUT_SWIGLU_BACKWARD, graph.cmdStream[0]);
        assertEquals(9, graph.cmdPos);
        assertEquals(5, graph.pendingAllocs.size());
        assertEquals(DType.FLOAT32, result.gateGradient().dtype());
        assertEquals(DType.FLOAT32, result.upGradient().dtype());
        assertArrayEquals(gate.shape(), result.gateGradient().shape());
    }

    @Test
    void adamRegistersEveryIndependentStateAndLogicalLayout() {
        Tensor w = new Tensor(2, 3, 4), g = new Tensor(2, 3, 4), m = new Tensor(2, 3, 4), v = new Tensor(2, 3, 4);
        graph.layoutAdamW(w, g, m, v, 0.1f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.1f, 0.001f);
        assertEquals(ComputeGraph.OP_LAYOUT_ADAMW, graph.cmdStream[0]);
        assertEquals(16, graph.cmdPos);
        assertEquals(4, graph.pendingAllocs.size());
        assertEquals(Float.floatToRawIntBits(0.1f), graph.cmdStream[9]);
    }

    @Test
    void emptyAndInvalidTrainingGeometryDoesNotAllocateValues() {
        Tensor empty = new Tensor(DType.FLOAT16, 0, 2, 3), gamma = new Tensor(DType.FLOAT16, 3);
        assertEquals(0, graph.layoutNormalize(empty, gamma, null, 1e-5f, false).output().numel());
        assertEquals(0, graph.layoutSwiGluBackward(new Tensor(empty.shape()), empty, empty).gateGradient().numel());
        assertThrows(IllegalArgumentException.class, () -> graph.layoutNormalize(empty, gamma, gamma, 1e-5f, false));
        assertThrows(IllegalArgumentException.class,
                () -> graph.layoutNormalize(empty, new Tensor(2), null, 1e-5f, false));
        assertThrows(IllegalArgumentException.class, () -> graph.layoutNormalizationBackward(
                empty, empty, new Tensor(1), gamma, false));
        assertTrue(graph.pendingAllocs.isEmpty());
    }
}
