package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

class MetalLegacyKernelTest extends MetalDifferentialTestSupport {

    @Test
    void nonAtomicScatterAccumulatesDistinctRowsCorrectly() {
        var graph = new ComputeGraph(new MetalGpuRuntime());
        Tensor target = Tensor.from2D(new float[][]{{10, 20}, {30, 40}, {50, 60}});
        Tensor indices = Tensor.from2D(new float[][]{{2}, {0}, {1}});
        Tensor gradient = Tensor.from2D(new float[][]{{1, 2}, {3, 4}, {5, 6}});
        try {
            var buffer = graph.ensureGpuBuffer(target);
            graph.recordScatterAddRows(buffer, graph.ensureGpuBuffer(indices),
                    graph.ensureGpuBuffer(gradient), 3, 2, 3);
            buffer.cpuStale = true;
            graph.materialize(target);
            assertTensorClose(Tensor.from2D(new float[][]{{13, 24}, {35, 46}, {51, 62}}), target, 1e-6f, 1e-6f);
        } finally {
            graph.releaseAll();
        }
    }

    @Test
    void rowSumSquaresMatchesExpectedValues() {
        var graph = new ComputeGraph(new MetalGpuRuntime());
        Tensor input = Tensor.from2D(new float[][]{{1, -2, 3}, {-4, 5, -6}});
        var output = graph.newOutputBuffer(2, 1);
        Tensor actual = graph.createOutputTensor(output);
        try {
            graph.recordSumSquares(graph.ensureGpuBuffer(input), output, 2, 3);
            graph.materialize(actual);
            assertTensorClose(Tensor.from2D(new float[][]{{14}, {77}}), actual, 1e-6f, 1e-6f);
        } finally {
            graph.releaseAll();
        }
    }

    @Test
    void nonCausalSoftmaxAndBackwardMatchCpuWithLargeLogits() {
        Tensor logits = Tensor.from2D(new float[][]{{1000, 1001, 999}, {-1000, -1002, -1001}});
        Tensor gradient = Tensor.from2D(new float[][]{{1, -2, 3}, {-4, 5, -6}});
        Tensor expected = cpu.softmaxRows(logits);
        Tensor actual = logits.softmaxRows();
        assertTensorClose(expected, actual, 1e-5f, 1e-5f);
        assertTensorClose(cpu.softmaxBackward(gradient, expected),
                gradient.softmaxBackward(actual), 1e-5f, 1e-5f);
    }
}
