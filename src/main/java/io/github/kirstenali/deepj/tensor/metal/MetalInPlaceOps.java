package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;

import java.util.List;

abstract class MetalInPlaceOps extends MetalMathOps {
    void bindInPlaceResult(Tensor target, GpuBuffer out) {
        graph.bindTensorToBuffer(target, out);
    }

    @Override
    public void addInPlace(Tensor a, Tensor b) {
        Tensor.requireSameShape(a, b, "addInPlace");
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordBinary(ComputeGraph.OP_ADD, ga, gb, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void subtractInPlace(Tensor a, Tensor b) {
        Tensor.requireSameShape(a, b, "subtractInPlace");
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordBinary(ComputeGraph.OP_SUBTRACT, ga, gb, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void multiplyInPlace(Tensor a, Tensor b) {
        Tensor.requireSameShape(a, b, "multiplyInPlace");
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordBinary(ComputeGraph.OP_MULTIPLY, ga, gb, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void divideInPlace(Tensor a, Tensor b) {
        Tensor.requireSameShape(a, b, "divideInPlace");
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordBinary(ComputeGraph.OP_DIVIDE, ga, gb, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void multiplyScalarInPlace(Tensor a, float s) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordMultiplyScalar(ga, gOut, s);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void addScalarInPlace(Tensor a, float s) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordScalarUnary(ComputeGraph.OP_ADD_SCALAR, ga, gOut, s);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void divideScalarInPlace(Tensor a, float s) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordScalarUnary(ComputeGraph.OP_DIVIDE_SCALAR, ga, gOut, s);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void sqrtInPlace(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_SQRT, ga, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void negInPlace(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_NEG, ga, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void expInPlace(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_EXP, ga, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void logInPlace(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_LOG, ga, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void reluInPlace(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_RELU, ga, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void geluInPlace(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_GELU, ga, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void tanhInPlace(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_TANH, ga, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void sigmoidInPlace(Tensor a) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_SIGMOID, ga, gOut);
        bindInPlaceResult(a, gOut);
    }


}
