package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;

/**
 * Routes large compact products onto MPS through lossless FLOAT32 staging. BF16 operands are
 * always widened; a FLOAT16 left operand is widened when the right operand is FLOAT32 because
 * MPS only mixes inputs as A=FLOAT32, B=FLOAT16. FLOAT32 x FLOAT16 and FLOAT16 x FLOAT16 go natively.
 */
final class MetalMixedMatmul {
    static final int MPS_FLOOR = 64;

    private MetalMixedMatmul() {}

    static Tensor product(ComputeGraph graph, Tensor left, Tensor right) {
        if (!stage(left, right)) return graph.layoutMatmul(left, right);
        try (var scope = TensorScope.open()) {
            Tensor result = graph.layoutMatmul(widenLeft(graph, left), widenRight(graph, right));
            return scope.escape(narrowResult(graph, result, left, right));
        }
    }

    static boolean stage(Tensor left, Tensor right) {
        boolean compact = hasBfloat(left, right)
                || (left.dtype() == DType.FLOAT16 && right.dtype() == DType.FLOAT32);
        return compact && left.rank() >= 2 && right.rank() >= 2 && largeProduct(left, right)
                && castsCheaperThanOutput(left, right);
    }

    static boolean stageBfloat(Tensor left, Tensor right) {
        return hasBfloat(left, right) && stage(left, right);
    }

    private static boolean hasBfloat(Tensor left, Tensor right) {
        return left.dtype() == DType.BFLOAT16 || right.dtype() == DType.BFLOAT16;
    }

    /** Keep products on the compact kernel when staging would widen a dominant operand. */
    private static boolean castsCheaperThanOutput(Tensor left, Tensor right) {
        long widened = (left.dtype() == DType.FLOAT32 ? 0 : left.numel())
                + (right.dtype() == DType.BFLOAT16 ? right.numel() : 0);
        long output = (long) left.numel() / left.size(-1) * right.size(-1);
        return widened <= 2 * output;
    }

    private static boolean largeProduct(Tensor left, Tensor right) {
        int rows = left.size(-2), inner = left.size(-1), columns = right.size(-1);
        return inner == right.size(-2) && rows >= MPS_FLOOR
                && columns >= MPS_FLOOR && inner >= MPS_FLOOR;
    }

    private static Tensor widenLeft(ComputeGraph graph, Tensor left) {
        return left.dtype() == DType.FLOAT32 ? left : graph.castToDevice(left, DType.FLOAT32);
    }

    private static Tensor widenRight(ComputeGraph graph, Tensor right) {
        return right.dtype() == DType.BFLOAT16 ? graph.castToDevice(right, DType.FLOAT32) : right;
    }

    private static Tensor narrowResult(ComputeGraph graph, Tensor result, Tensor left, Tensor right) {
        boolean bothBfloat = left.dtype() == DType.BFLOAT16 && right.dtype() == DType.BFLOAT16;
        return bothBfloat ? graph.castToDevice(result, DType.BFLOAT16) : result;
    }
}
