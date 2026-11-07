package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Eligible BF16 matrices stay compact when MPSGraph can handle their physical layout. */
final class MetalCompactGraph {
    private MetalCompactGraph() {}

    static boolean supports(Tensor left, Tensor right) {
        return left.dtype() == DType.BFLOAT16 && right.dtype() == DType.BFLOAT16
                && matrix(left) && matrix(right) && dimensions(left, right) && Capability.AVAILABLE;
    }

    private static boolean dimensions(Tensor left, Tensor right) {
        return (left.size(-2) == 1 || left.size(-2) >= MetalMixedMatmul.MPS_FLOOR)
                && left.size(-1) == right.size(-2) && left.size(-1) >= 256
                && right.size(-1) >= 256 && right.size(-1) % 8 == 0;
    }

    private static boolean matrix(Tensor value) {
        if (value.rank() < 2 || value.numel() != (long) value.size(-2) * value.size(-1)) return false;
        long[] strides = value.strides();
        long row = strides[strides.length - 2], column = strides[strides.length - 1];
        if (value.storageOffset() % 8 != 0) return false;
        return packed(column, row, value.size(-1)) || packed(row, column, value.size(-2));
    }

    private static boolean packed(long inner, long outer, int width) {
        return inner == 1 && outer == width && outer % 8 == 0;
    }

    private static final class Capability {
        private static final boolean AVAILABLE = MetalNative.AVAILABLE && MetalNative.nativeSupportsCompactGraph();
        private Capability() {}
    }
}
