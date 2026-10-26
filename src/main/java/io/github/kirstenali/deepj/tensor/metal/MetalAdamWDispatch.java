package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;

/** Element-count fused Adam is rank independent; partial/strided views retain their layout descriptors. */
final class MetalAdamWDispatch {
    private MetalAdamWDispatch() {}

    static void update(ComputeGraph graph, Tensor w, Tensor g, Tensor m, Tensor v, float lr, float beta1,
                        float beta2, float epsilon, float decay, float bc1, float bc2) {
        if (!MetalDenseOps.dense(w) || !MetalDenseOps.dense(g) || !MetalDenseOps.dense(m) || !MetalDenseOps.dense(v)) {
            graph.layoutAdamW(w, g, m, v, lr, beta1, beta2, epsilon, decay, bc1, bc2);
            return;
        }
        GpuBuffer weight = graph.ensureStorageBuffer(w), gradient = graph.ensureStorageBuffer(g);
        GpuBuffer first = graph.ensureStorageBuffer(m), second = graph.ensureStorageBuffer(v);
        graph.recordAdamWUpdate(weight, gradient, first, second, lr, beta1, beta2, epsilon, decay, bc1, bc2, w.numel());
        graph.markUpdated(w);
        graph.markUpdated(m);
        graph.markUpdated(v);
    }
}
