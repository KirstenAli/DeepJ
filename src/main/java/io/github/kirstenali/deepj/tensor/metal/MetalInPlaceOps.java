package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;

import java.util.List;

abstract class MetalInPlaceOps extends MetalMathOps {
    void bindInPlaceResult(Tensor target, GpuBuffer out) {
        graph.bindTensorToBuffer(target, out);
    }

    @Override
    public void addInPlace(Tensor a, Tensor b) {
        if (MetalLayoutInPlace.binary(graph, a, b, ComputeGraph.OP_ADD)) return;
        Tensor.requireSameShape(a, b, "addInPlace");
        if (MetalHostZeroInput.canSkip(a, b)) {
            addToHostZero(a, b);
            return;
        }
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordBinary(ComputeGraph.OP_ADD, ga, gb, gOut);
        bindInPlaceResult(a, gOut);
    }

    private void addToHostZero(Tensor target, Tensor source) {
        GpuBuffer output = graph.newOutputBuffer(target.matrixRows(), target.matrixCols());
        graph.recordScalarUnary(ComputeGraph.OP_ADD_SCALAR, gpuIn(source), output, 0.0f);
        bindInPlaceResult(target, output);
    }

    @Override
    public void subtractInPlace(Tensor a, Tensor b) {
        if (MetalLayoutInPlace.binary(graph, a, b, ComputeGraph.OP_SUBTRACT)) return;
        Tensor.requireSameShape(a, b, "subtractInPlace");
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordBinary(ComputeGraph.OP_SUBTRACT, ga, gb, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void multiplyInPlace(Tensor a, Tensor b) {
        if (MetalLayoutInPlace.binary(graph, a, b, ComputeGraph.OP_MULTIPLY)) return;
        Tensor.requireSameShape(a, b, "multiplyInPlace");
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordBinary(ComputeGraph.OP_MULTIPLY, ga, gb, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void divideInPlace(Tensor a, Tensor b) {
        if (MetalLayoutInPlace.binary(graph, a, b, ComputeGraph.OP_DIVIDE)) return;
        Tensor.requireSameShape(a, b, "divideInPlace");
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordBinary(ComputeGraph.OP_DIVIDE, ga, gb, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void multiplyScalarInPlace(Tensor a, float s) {
        if (MetalLayoutInPlace.unary(graph, a, ComputeGraph.OP_MULTIPLY_SCALAR, s)) return;
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordMultiplyScalar(ga, gOut, s);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void addScalarInPlace(Tensor a, float s) {
        if (MetalLayoutInPlace.unary(graph, a, ComputeGraph.OP_ADD_SCALAR, s)) return;
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordScalarUnary(ComputeGraph.OP_ADD_SCALAR, ga, gOut, s);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void divideScalarInPlace(Tensor a, float s) {
        if (MetalLayoutInPlace.unary(graph, a, ComputeGraph.OP_DIVIDE_SCALAR, s)) return;
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordScalarUnary(ComputeGraph.OP_DIVIDE_SCALAR, ga, gOut, s);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void sqrtInPlace(Tensor a) {
        if (MetalLayoutInPlace.unary(graph, a, ComputeGraph.OP_SQRT, 0)) return;
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_SQRT, ga, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void negInPlace(Tensor a) {
        if (MetalLayoutInPlace.unary(graph, a, ComputeGraph.OP_NEG, 0)) return;
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_NEG, ga, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void expInPlace(Tensor a) {
        if (MetalLayoutInPlace.unary(graph, a, ComputeGraph.OP_EXP, 0)) return;
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_EXP, ga, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void logInPlace(Tensor a) {
        if (MetalLayoutInPlace.unary(graph, a, ComputeGraph.OP_LOG, 0)) return;
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_LOG, ga, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void reluInPlace(Tensor a) {
        if (MetalLayoutInPlace.unary(graph, a, ComputeGraph.OP_RELU, 0)) return;
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_RELU, ga, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void geluInPlace(Tensor a) {
        if (MetalLayoutInPlace.unary(graph, a, ComputeGraph.OP_GELU, 0)) return;
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_GELU, ga, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void tanhInPlace(Tensor a) {
        if (MetalLayoutInPlace.unary(graph, a, ComputeGraph.OP_TANH, 0)) return;
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_TANH, ga, gOut);
        bindInPlaceResult(a, gOut);
    }

    @Override
    public void sigmoidInPlace(Tensor a) {
        if (MetalLayoutInPlace.unary(graph, a, ComputeGraph.OP_SIGMOID, 0)) return;
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordUnary(ComputeGraph.OP_SIGMOID, ga, gOut);
        bindInPlaceResult(a, gOut);
    }


}
