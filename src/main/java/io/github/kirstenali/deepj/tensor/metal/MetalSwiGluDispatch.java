package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;

/** Shape-independent FLOAT32 kernels remain fused; typed and strided inputs use layout-aware kernels. */
final class MetalSwiGluDispatch {
    private MetalSwiGluDispatch() {}

    static Tensor forward(ComputeGraph graph, Tensor gate, Tensor up) {
        if (!MetalDenseOps.dense(gate) || !MetalDenseOps.dense(up)) {
            return graph.layoutBinary(gate, up, ComputeGraph.OP_SWIGLU);
        }
        GpuBuffer first = graph.ensureStorageBuffer(gate), second = graph.ensureStorageBuffer(up);
        GpuBuffer output = graph.newOutputBuffer(1, gate.numel());
        graph.recordSwiGlu(first, second, output);
        return graph.createOutputLike(output, gate);
    }

    static SwiGluBackwardResult backward(ComputeGraph graph, Tensor gradient, Tensor gate, Tensor up) {
        if (!MetalDenseOps.dense(gate) || !MetalDenseOps.dense(up) || !MetalDenseOps.dense(gradient)) {
            return graph.layoutSwiGluBackward(gradient, gate, up);
        }
        GpuBuffer g = graph.ensureStorageBuffer(gradient), a = graph.ensureStorageBuffer(gate);
        GpuBuffer b = graph.ensureStorageBuffer(up);
        GpuBuffer first = graph.newOutputBuffer(1, gate.numel()), second = graph.newOutputBuffer(1, up.numel());
        graph.recordSwiGluBackward(g, a, b, first, second);
        return new SwiGluBackwardResult(graph.createOutputLike(first, gate),
                graph.createOutputLike(second, up));
    }
}
