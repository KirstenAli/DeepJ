package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.*;

/** Short compact cache prefixes share one inference dispatch without global score/probability buffers. */
final class MetalCachedAttention {
    private MetalCachedAttention() {}

    static boolean supports(Tensor q, Tensor k, Tensor v, int heads, int length, float scale) {
        return !GradMode.isEnabled() && types(q, k, v) && layouts(q, k, v)
                && shape(q, k, v, heads) && limits(q, k, heads, length, scale);
    }

    private static boolean types(Tensor q, Tensor k, Tensor v) {
        return q.dtype() != DType.FLOAT32 && q.dtype() == k.dtype() && q.dtype() == v.dtype();
    }

    private static boolean layouts(Tensor q, Tensor k, Tensor v) { return dense(q) && dense(k) && dense(v); }

    private static boolean dense(Tensor tensor) {
        return tensor.rank() == 2 && tensor.isContiguous() && tensor.storageOffset() == 0;
    }

    private static boolean shape(Tensor q, Tensor k, Tensor v, int heads) {
        if (heads < 1) return false;
        return q.matrixRows() > 0 && q.matrixRows() % heads == 0 && k.matrixRows() % heads == 0
                && k.hasSameShape(v) && q.matrixCols() == k.matrixCols();
    }

    private static boolean limits(Tensor q, Tensor k, int heads, int length, float scale) {
        return q.matrixCols() > 0 && q.matrixCols() <= 128 && length > 0 && length <= 512
                && length <= k.matrixRows() / heads && Float.isFinite(scale);
    }

    static Tensor apply(ComputeGraph graph, Tensor q, Tensor k, Tensor v, int heads, int length, float scale) {
        if (!supports(q, k, v, heads, length, scale)) {
            throw new IllegalArgumentException("Unsupported cached attention");
        }
        GpuBuffer output = graph.newOutputBuffer(q.matrixRows(), q.matrixCols(), q.dtype());
        graph.recordCachedAttention(graph.ensureStorageBuffer(q), graph.ensureStorageBuffer(k),
                graph.ensureStorageBuffer(v), output, q.matrixRows(), q.matrixCols(), heads,
                k.matrixRows() / heads, length, scale);
        return graph.createOutputLike(output, q);
    }
}
