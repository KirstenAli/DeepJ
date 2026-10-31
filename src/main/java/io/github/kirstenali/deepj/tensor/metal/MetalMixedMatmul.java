package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;

/** Bounded per-product widening for large narrow mixed BF16 backward products. */
final class MetalMixedMatmul {
    private MetalMixedMatmul() {}

    static Tensor product(ComputeGraph graph, Tensor left, Tensor right) {
        if (!widenRight(left, right)) return graph.layoutMatmul(left, right);
        try (var scope = TensorScope.open()) {
            Tensor widened = graph.castToDevice(right, DType.FLOAT32);
            return scope.escape(graph.layoutMatmul(left, widened));
        }
    }

    static boolean widenRight(Tensor left, Tensor right) {
        if (left.dtype() != DType.FLOAT32 || right.dtype() != DType.BFLOAT16
                || left.rank() < 2 || right.rank() < 2) return false;
        return narrowProduct(left, right);
    }

    private static boolean narrowProduct(Tensor left, Tensor right) {
        int rows = left.size(-2), inner = left.size(-1), columns = right.size(-1);
        return inner == right.size(-2) && rows >= 1024 && columns >= 64 && inner >= 64
                && (columns == 64 || inner == 64);
    }
}
