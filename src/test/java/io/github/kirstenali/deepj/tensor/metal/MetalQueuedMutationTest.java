package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class MetalQueuedMutationTest extends MetalDifferentialTestSupport {
    @Test
    void hostEditCannotOverwriteTheSnapshotUsedByAnEarlierQueuedConsumer() {
        Tensor input = Tensor.from2D(new float[][]{{1, 2}});
        var source = metal.graph.ensureGpuBuffer(input);
        var first = metal.graph.newOutputBuffer(1, 2);
        Tensor earlier = metal.graph.createOutputTensor(first);
        metal.graph.recordUnary(ComputeGraph.OP_NEG, source, first);
        input.set(0, 0, 9);
        var second = metal.graph.newOutputBuffer(1, 2);
        Tensor later = metal.graph.createOutputTensor(second);
        metal.graph.recordUnary(ComputeGraph.OP_NEG, metal.graph.ensureGpuBuffer(input), second);
        assertArrayEquals(new float[]{-9, -2}, later.toFloatArray());
        assertArrayEquals(new float[]{-1, -2}, earlier.toFloatArray());
    }
}
