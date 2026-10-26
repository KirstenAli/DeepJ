package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;

import java.util.List;

abstract class MetalMathOps extends MetalCoreOps {
    @Override
    public Tensor multiplyScalar(Tensor a, float scalar) {
        if (!MetalDenseOps.matrix(a)) return MetalDenseOps.scalar(graph, a, ComputeGraph.OP_MULTIPLY_SCALAR, scalar);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordMultiplyScalar(ga, gOut, scalar);
        return gpuOut(gOut);
    }

    @Override
    public Tensor addScalar(Tensor a, float scalar) {
        if (!MetalDenseOps.matrix(a)) return MetalDenseOps.scalar(graph, a, ComputeGraph.OP_ADD_SCALAR, scalar);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordScalarUnary(ComputeGraph.OP_ADD_SCALAR, ga, gOut, scalar);
        return gpuOut(gOut);
    }

    @Override
    public Tensor divideScalar(Tensor a, float scalar) {
        if (!MetalDenseOps.matrix(a)) return MetalDenseOps.scalar(graph, a, ComputeGraph.OP_DIVIDE_SCALAR, scalar);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordScalarUnary(ComputeGraph.OP_DIVIDE_SCALAR, ga, gOut, scalar);
        return gpuOut(gOut);
    }

    @Override
    public Tensor sumRows(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(1, a.matrixCols());
        graph.recordReduction(ComputeGraph.OP_SUM_ROWS, ga, gOut, a.matrixRows(), a.matrixCols());
        return gpuOut(gOut);
    }

    @Override
    public Tensor sumAlongRows(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), 1);
        graph.recordReduction(ComputeGraph.OP_SUM_ALONG_ROWS, ga, gOut, a.matrixRows(), a.matrixCols());
        return gpuOut(gOut);
    }

    @Override
    public Tensor sumAlongCols(Tensor a) {
        return sumRows(a);
    }

    @Override
    public Tensor meanAlongRows(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), 1);
        graph.recordReduction(ComputeGraph.OP_MEAN_ALONG_ROWS, ga, gOut, a.matrixRows(), a.matrixCols());
        return gpuOut(gOut);
    }

    @Override
    public Tensor varianceAlongRows(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), 1);
        graph.recordReduction(ComputeGraph.OP_VARIANCE_ALONG_ROWS, ga, gOut, a.matrixRows(), a.matrixCols());
        return gpuOut(gOut);
    }

    @Override
    public Tensor maxAlongRows(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), 1);
        graph.recordReduction(ComputeGraph.OP_MAX_ALONG_ROWS, ga, gOut, a.matrixRows(), a.matrixCols());
        return gpuOut(gOut);
    }

    @Override
    public float sum(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(1, 1);
        graph.recordSumScalar(ga, gOut, a.matrixRows(), a.matrixCols());
        Tensor scalar = gpuOut(gOut);
        scalar.materialize();
        return scalar.rawData()[0];
    }

    @Override
    public float sumAbs(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gRowAbsSums = graph.newOutputBuffer(a.matrixRows(), 1);
        graph.recordSumAbs(ga, gRowAbsSums, a.matrixRows(), a.matrixCols());
        Tensor rowAbsSums = gpuOut(gRowAbsSums);
        Tensor scalar = sumRows(rowAbsSums);
        scalar.materialize();
        return scalar.rawData()[0];
    }

    @Override
    public float l2Norm(List<Tensor> tensors) {
        if (tensors.isEmpty()) return 0.0f;
        Tensor total = new Tensor(1, 1);
        GpuBuffer output = gpuIn(total);
        for (Tensor tensor : tensors) {
            graph.recordSumSquaresScalar(gpuIn(tensor), output);
        }
        output.cpuStale = true;
        total.materialize();
        return (float) Math.sqrt(total.rawData()[0]);
    }

    @Override
    public Tensor transpose(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixCols(), a.matrixRows());
        graph.recordTranspose(ga, gOut, a.matrixRows(), a.matrixCols());
        return gpuOut(gOut);
    }

    @Override
    public Tensor clamp(Tensor a, float min, float max) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutUnary(a, ComputeGraph.OP_CLAMP, min, max);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordClamp(ga, gOut, min, max);
        return gpuOut(gOut);
    }

    @Override
    public Tensor pow(Tensor a, float exponent) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutUnary(a, ComputeGraph.OP_POW, exponent, 0);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordPow(ga, gOut, exponent);
        return gpuOut(gOut);
    }

    @Override
    public Tensor sqrt(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutUnary(a, ComputeGraph.OP_SQRT, 0, 0);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_SQRT, ga, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor neg(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutUnary(a, ComputeGraph.OP_NEG, 0, 0);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_NEG, ga, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor exp(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutUnary(a, ComputeGraph.OP_EXP, 0, 0);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_EXP, ga, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor log(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutUnary(a, ComputeGraph.OP_LOG, 0, 0);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_LOG, ga, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor tanh(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutUnary(a, ComputeGraph.OP_TANH, 0, 0);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_TANH, ga, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor sigmoid(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutUnary(a, ComputeGraph.OP_SIGMOID, 0, 0);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_SIGMOID, ga, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor relu(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutUnary(a, ComputeGraph.OP_RELU, 0, 0);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_RELU, ga, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor reluBackward(Tensor input, Tensor gradOutput) {
        if (!MetalDenseOps.sameMatrices(input, gradOutput)) {
            Tensor.requireSameShape(input, gradOutput, "reluBackward");
            return graph.layoutBinary(input, gradOutput, ComputeGraph.OP_RELU_BACKWARD);
        }
        GpuBuffer gi = gpuIn(input), gg = gpuIn(gradOutput);
        GpuBuffer gOut = graph.newOutputBuffer(input.matrixRows(), input.matrixCols());
        graph.recordBinary(ComputeGraph.OP_RELU_BACKWARD, gi, gg, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor gelu(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutUnary(a, ComputeGraph.OP_GELU, 0, 0);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_GELU, ga, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor geluBackward(Tensor input, Tensor gradOutput) {
        if (!MetalDenseOps.sameMatrices(input, gradOutput)) {
            Tensor.requireSameShape(input, gradOutput, "geluBackward");
            return graph.layoutBinary(input, gradOutput, ComputeGraph.OP_GELU_BACKWARD);
        }
        GpuBuffer gi = gpuIn(input), gg = gpuIn(gradOutput);
        GpuBuffer gOut = graph.newOutputBuffer(input.matrixRows(), input.matrixCols());
        graph.recordBinary(ComputeGraph.OP_GELU_BACKWARD, gi, gg, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor softmaxRows(Tensor logits) {
        GpuBuffer ga = gpuIn(logits);
        GpuBuffer gOut = graph.newOutputBuffer(logits.matrixRows(), logits.matrixCols());
        graph.recordSoftmaxRows(ga, gOut, logits.matrixRows(), logits.matrixCols());
        return gpuOut(gOut);
    }

    @Override
    public Tensor softmaxBackward(Tensor gradOutput, Tensor softmaxOut) {
        Tensor.requireSameShape(gradOutput, softmaxOut, "softmaxBackward");
        GpuBuffer gGrad = gpuIn(gradOutput);
        GpuBuffer gSoftmax = gpuIn(softmaxOut);
        GpuBuffer gOut = graph.newOutputBuffer(gradOutput.matrixRows(), gradOutput.matrixCols());
        graph.recordSoftmaxBackward(gGrad, gSoftmax, gOut, gradOutput.matrixRows(), gradOutput.matrixCols());
        return gpuOut(gOut);
    }

    @Override
    public Tensor softmaxBackward(Tensor gradient, Tensor probabilities, float scale) {
        Tensor.requireSameShape(gradient, probabilities, "softmaxBackward");
        GpuBuffer output = graph.newOutputBuffer(gradient.matrixRows(), gradient.matrixCols());
        graph.recordScaledSoftmaxBackward(gpuIn(gradient), gpuIn(probabilities), output, scale);
        return gpuOut(output);
    }


}
