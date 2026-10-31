package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;

abstract class MetalMathOps extends MetalRowReductionOps {
    @Override
    public Tensor multiplyScalar(Tensor a, float scalar) {
        if (!MetalDenseOps.matrix(a)) return MetalDenseOps.scalar(graph, a, ComputeGraph.OP_MULTIPLY_SCALAR, scalar);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordMultiplyScalar(ga, gOut, scalar);
        return graph.createOutputLike(gOut, a);
    }

    @Override
    public Tensor addScalar(Tensor a, float scalar) {
        if (!MetalDenseOps.matrix(a)) return MetalDenseOps.scalar(graph, a, ComputeGraph.OP_ADD_SCALAR, scalar);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordScalarUnary(ComputeGraph.OP_ADD_SCALAR, ga, gOut, scalar);
        return graph.createOutputLike(gOut, a);
    }

    @Override
    public Tensor divideScalar(Tensor a, float scalar) {
        if (!MetalDenseOps.matrix(a)) return MetalDenseOps.scalar(graph, a, ComputeGraph.OP_DIVIDE_SCALAR, scalar);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordScalarUnary(ComputeGraph.OP_DIVIDE_SCALAR, ga, gOut, scalar);
        return graph.createOutputLike(gOut, a);
    }


    @Override
    public Tensor clamp(Tensor a, float min, float max) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutUnary(a, ComputeGraph.OP_CLAMP, min, max);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordClamp(ga, gOut, min, max);
        return graph.createOutputLike(gOut, a);
    }

    @Override
    public Tensor pow(Tensor a, float exponent) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutUnary(a, ComputeGraph.OP_POW, exponent, 0);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordPow(ga, gOut, exponent);
        return graph.createOutputLike(gOut, a);
    }

    @Override
    public Tensor sqrt(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutUnary(a, ComputeGraph.OP_SQRT, 0, 0);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_SQRT, ga, gOut);
        return graph.createOutputLike(gOut, a);
    }

    @Override
    public Tensor neg(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutUnary(a, ComputeGraph.OP_NEG, 0, 0);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_NEG, ga, gOut);
        return graph.createOutputLike(gOut, a);
    }

    @Override
    public Tensor exp(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutUnary(a, ComputeGraph.OP_EXP, 0, 0);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_EXP, ga, gOut);
        return graph.createOutputLike(gOut, a);
    }

    @Override
    public Tensor log(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutUnary(a, ComputeGraph.OP_LOG, 0, 0);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_LOG, ga, gOut);
        return graph.createOutputLike(gOut, a);
    }

    @Override
    public Tensor tanh(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutUnary(a, ComputeGraph.OP_TANH, 0, 0);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_TANH, ga, gOut);
        return graph.createOutputLike(gOut, a);
    }

    @Override
    public Tensor sigmoid(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutUnary(a, ComputeGraph.OP_SIGMOID, 0, 0);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_SIGMOID, ga, gOut);
        return graph.createOutputLike(gOut, a);
    }

    @Override
    public Tensor relu(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutUnary(a, ComputeGraph.OP_RELU, 0, 0);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_RELU, ga, gOut);
        return graph.createOutputLike(gOut, a);
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
        return graph.createOutputLike(gOut, input);
    }

    @Override
    public Tensor gelu(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutUnary(a, ComputeGraph.OP_GELU, 0, 0);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_GELU, ga, gOut);
        return graph.createOutputLike(gOut, a);
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
        return graph.createOutputLike(gOut, input);
    }

    @Override
    public Tensor softmaxRows(Tensor logits) {
        MatrixShapeChecks.requireMatrix(logits);
        if (!MetalDenseOps.matrix(logits)) return graph.layoutSoftmax(logits, 1);
        GpuBuffer ga = gpuIn(logits);
        GpuBuffer gOut = graph.newOutputBuffer(logits.matrixRows(), logits.matrixCols());
        graph.recordSoftmaxRows(ga, gOut, logits.matrixRows(), logits.matrixCols());
        return graph.createOutputLike(gOut, logits);
    }

    @Override
    public Tensor softmaxBackward(Tensor gradOutput, Tensor softmaxOut) {
        Tensor.requireSameShape(gradOutput, softmaxOut, "softmaxBackward");
        if (!MetalDenseOps.sameMatrices(gradOutput, softmaxOut)) {
            MatrixShapeChecks.requireMatrix(softmaxOut);
            return axisSoftmaxBackward(gradOutput, softmaxOut, 1);
        }
        GpuBuffer gGrad = gpuIn(gradOutput);
        GpuBuffer gSoftmax = gpuIn(softmaxOut);
        GpuBuffer gOut = graph.newOutputBuffer(gradOutput.matrixRows(), gradOutput.matrixCols());
        graph.recordSoftmaxBackward(gGrad, gSoftmax, gOut, gradOutput.matrixRows(), gradOutput.matrixCols());
        return graph.createOutputLike(gOut, gradOutput);
    }

    @Override
    public Tensor softmaxBackward(Tensor gradient, Tensor probabilities, float scale) {
        Tensor.requireSameShape(gradient, probabilities, "softmaxBackward");
        if (!MetalDenseOps.sameMatrices(gradient, probabilities)) {
            MatrixShapeChecks.requireMatrix(probabilities);
            return multiplyScalar(axisSoftmaxBackward(gradient, probabilities, 1), scale);
        }
        GpuBuffer output = graph.newOutputBuffer(gradient.matrixRows(), gradient.matrixCols());
        graph.recordScaledSoftmaxBackward(gpuIn(gradient), gpuIn(probabilities), output, scale);
        return graph.createOutputLike(output, gradient);
    }


}
