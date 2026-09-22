package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;

import java.util.List;

abstract class MetalMathOps extends MetalCoreOps {
    @Override
    public Tensor multiplyScalar(Tensor a, float scalar) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordMultiplyScalar(ga, gOut, scalar);
        return gpuOut(gOut);
    }

    @Override
    public Tensor addScalar(Tensor a, float scalar) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordScalarUnary(ComputeGraph.OP_ADD_SCALAR, ga, gOut, scalar);
        return gpuOut(gOut);
    }

    @Override
    public Tensor divideScalar(Tensor a, float scalar) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordScalarUnary(ComputeGraph.OP_DIVIDE_SCALAR, ga, gOut, scalar);
        return gpuOut(gOut);
    }

    @Override
    public Tensor sumRows(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(1, a.cols);
        graph.recordReduction(ComputeGraph.OP_SUM_ROWS, ga, gOut, a.rows, a.cols);
        return gpuOut(gOut);
    }

    @Override
    public Tensor sumAlongRows(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, 1);
        graph.recordReduction(ComputeGraph.OP_SUM_ALONG_ROWS, ga, gOut, a.rows, a.cols);
        return gpuOut(gOut);
    }

    @Override
    public Tensor sumAlongCols(Tensor a) {
        return sumRows(a);
    }

    @Override
    public Tensor meanAlongRows(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, 1);
        graph.recordReduction(ComputeGraph.OP_MEAN_ALONG_ROWS, ga, gOut, a.rows, a.cols);
        return gpuOut(gOut);
    }

    @Override
    public Tensor varianceAlongRows(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, 1);
        graph.recordReduction(ComputeGraph.OP_VARIANCE_ALONG_ROWS, ga, gOut, a.rows, a.cols);
        return gpuOut(gOut);
    }

    @Override
    public Tensor maxAlongRows(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, 1);
        graph.recordReduction(ComputeGraph.OP_MAX_ALONG_ROWS, ga, gOut, a.rows, a.cols);
        return gpuOut(gOut);
    }

    @Override
    public float sum(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(1, 1);
        graph.recordSumScalar(ga, gOut, a.rows, a.cols);
        Tensor scalar = gpuOut(gOut);
        scalar.materialize();
        return scalar.data[0];
    }

    @Override
    public float sumAbs(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gRowAbsSums = graph.newOutputBuffer(a.rows, 1);
        graph.recordSumAbs(ga, gRowAbsSums, a.rows, a.cols);
        Tensor rowAbsSums = gpuOut(gRowAbsSums);
        Tensor scalar = sumRows(rowAbsSums);
        scalar.materialize();
        return scalar.data[0];
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
        return (float) Math.sqrt(total.data[0]);
    }

    @Override
    public Tensor transpose(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.cols, a.rows);
        graph.recordTranspose(ga, gOut, a.rows, a.cols);
        return gpuOut(gOut);
    }

    @Override
    public Tensor clamp(Tensor a, float min, float max) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordClamp(ga, gOut, min, max);
        return gpuOut(gOut);
    }

    @Override
    public Tensor pow(Tensor a, float exponent) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordPow(ga, gOut, exponent);
        return gpuOut(gOut);
    }

    @Override
    public Tensor sqrt(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_SQRT, ga, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor neg(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_NEG, ga, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor exp(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_EXP, ga, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor log(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_LOG, ga, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor tanh(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_TANH, ga, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor sigmoid(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_SIGMOID, ga, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor relu(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_RELU, ga, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor reluBackward(Tensor input, Tensor gradOutput) {
        GpuBuffer gi = gpuIn(input), gg = gpuIn(gradOutput);
        GpuBuffer gOut = graph.newOutputBuffer(input.rows, input.cols);
        graph.recordBinary(ComputeGraph.OP_RELU_BACKWARD, gi, gg, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor gelu(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_GELU, ga, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor geluBackward(Tensor input, Tensor gradOutput) {
        GpuBuffer gi = gpuIn(input), gg = gpuIn(gradOutput);
        GpuBuffer gOut = graph.newOutputBuffer(input.rows, input.cols);
        graph.recordBinary(ComputeGraph.OP_GELU_BACKWARD, gi, gg, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor softmaxRows(Tensor logits) {
        GpuBuffer ga = gpuIn(logits);
        GpuBuffer gOut = graph.newOutputBuffer(logits.rows, logits.cols);
        graph.recordSoftmaxRows(ga, gOut, logits.rows, logits.cols);
        return gpuOut(gOut);
    }

    @Override
    public Tensor softmaxBackward(Tensor gradOutput, Tensor softmaxOut) {
        Tensor.requireSameShape(gradOutput, softmaxOut, "softmaxBackward");
        GpuBuffer gGrad = gpuIn(gradOutput);
        GpuBuffer gSoftmax = gpuIn(softmaxOut);
        GpuBuffer gOut = graph.newOutputBuffer(gradOutput.rows, gradOutput.cols);
        graph.recordSoftmaxBackward(gGrad, gSoftmax, gOut, gradOutput.rows, gradOutput.cols);
        return gpuOut(gOut);
    }


}
