package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Non-matrix mutations update shared physical storage instead of rebinding a view to a dense result. */
final class MetalLayoutInPlace {
    private MetalLayoutInPlace() {}

    static boolean binary(ComputeGraph graph, Tensor target, Tensor operand, int operation) {
        if (MetalDenseOps.sameMatrices(target, operand)) return false;
        graph.layoutBinaryInto(target, operand, operation);
        return true;
    }

    static boolean unary(ComputeGraph graph, Tensor target, int operation, float scalar) {
        if (MetalDenseOps.matrix(target)) return false;
        graph.layoutUnaryInto(target, operation, scalar);
        return true;
    }
}
