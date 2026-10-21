package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

class MetalLegacyKernelTest extends MetalDifferentialTestSupport {

    @Test
    void nonAtomicScatterAccumulatesDuplicateRowsCorrectly() {
        var graph = new ComputeGraph(new MetalGpuRuntime());
        Tensor target = new Tensor(3, 2);
        Tensor indices = Tensor.from2D(new float[][]{{0}, {2}, {0}});
        Tensor gradient = Tensor.from2D(new float[][]{{1, 2}, {3, 4}, {5, 6}});
        try {
            var buffer = graph.ensureGpuBuffer(target);
            graph.recordScatterAddRows(buffer, graph.ensureGpuBuffer(indices),
                    graph.ensureGpuBuffer(gradient), 3, 2, 3);
            buffer.cpuStale = true;
            graph.materialize(target);
            assertTensorClose(Tensor.from2D(new float[][]{{6, 8}, {0, 0}, {3, 4}}), target, 1e-6f, 1e-6f);
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
}
