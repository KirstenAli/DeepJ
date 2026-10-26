package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.MatrixShapeChecks;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.List;

final class MetalNdReductions {
    private MetalNdReductions() {}

    static Tensor matrix(ComputeGraph graph, Tensor input, int axis, int operation) {
        MatrixShapeChecks.requireMatrix(input);
        return graph.layoutReduction(input, new int[]{axis}, true, operation, 0);
    }

    static float norm(ComputeGraph graph, List<Tensor> tensors) {
        Tensor total = graph.zeroToDevice(DType.FLOAT32, new int[0]);
        for (Tensor tensor : tensors) {
            total = graph.layoutBinary(total, graph.layoutTotal(tensor, 5), ComputeGraph.OP_ADD);
        }
        return (float) Math.sqrt(total.item());
    }
}
