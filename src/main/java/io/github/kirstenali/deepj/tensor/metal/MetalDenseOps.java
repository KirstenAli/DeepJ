package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.GpuBuffer;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Reuses element-count kernels only for operations that are independent of logical axes. */
final class MetalDenseOps {
    private MetalDenseOps() {}

    static boolean matrix(Tensor input) {
        return input.rank() == 2 && input.numel() > 0 && input.dtype() == DType.FLOAT32 && full(input);
    }

    static boolean sameMatrices(Tensor left, Tensor right) {
        return matrix(left) && matrix(right) && left.hasSameShape(right);
    }

    private static boolean full(Tensor input) {
        return input.isContiguous() && input.storageOffset() == 0
                && input.storageBytes() == input.dtype().storageBytes(input.numel());
    }

    static Tensor scalar(ComputeGraph graph, Tensor input, int opcode, float scalar) {
        if (input.dtype() != DType.FLOAT32 || !full(input)) return graph.layoutUnary(input, opcode, scalar, 0);
        if (input.numel() == 0) return graph.copyToDevice(input);
        GpuBuffer source = graph.ensureStorageBuffer(input);
        int rows = input.rank() == 2 ? input.matrixRows() : 1;
        GpuBuffer output = graph.newOutputBuffer(rows, input.numel() / rows);
        graph.recordScalarUnary(opcode, source, output, scalar);
        return graph.createOutputTensor(output, input.shape());
    }
}
