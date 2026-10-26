package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;


abstract class MetalTrainingOps extends MetalLossOps {

    @Override
    public void adamWUpdate(Tensor w, Tensor g, Tensor mt, Tensor vt,
                            float lr, float beta1, float beta2, float eps,
                            float weightDecay, float bc1, float bc2) {
        OptimizerChecks.requireState(w, g, mt, vt);
        if (!MetalDenseOps.sameMatrices(w, g) || !MetalDenseOps.sameMatrices(mt, vt)) {
            MetalAdamWDispatch.update(graph, w, g, mt, vt, lr, beta1, beta2, eps, weightDecay, bc1, bc2);
            return;
        }

        GpuBuffer gw = gpuIn(w);
        GpuBuffer gg = gpuIn(g);
        GpuBuffer gmt = gpuIn(mt);
        GpuBuffer gvt = gpuIn(vt);
        graph.recordAdamWUpdate(
                gw, gg, gmt, gvt,
                lr, beta1, beta2, eps,
                weightDecay, bc1, bc2,
                w.matrixRows() * w.matrixCols()
        );

        graph.markUpdated(w);
        graph.markUpdated(mt);
        graph.markUpdated(vt);
    }

    @Override
    public Tensor swiGlu(Tensor gate, Tensor up) {
        Tensor.requireSameShape(gate, up, "swiGlu");
        if (!MetalDenseOps.sameMatrices(gate, up)) return MetalSwiGluDispatch.forward(graph, gate, up);
        GpuBuffer fused = graph.newOutputBuffer(gate.matrixRows(), gate.matrixCols());
        graph.recordSwiGlu(gpuIn(gate), gpuIn(up), fused);
        return gpuOut(fused);
    }

    @Override
    public SwiGluBackwardResult swiGluBackward(Tensor gradient, Tensor gate, Tensor up) {
        Tensor.requireSameShape(gradient, gate, "swiGluBackward");
        Tensor.requireSameShape(gate, up, "swiGluBackward");
        if (!MetalDenseOps.sameMatrices(gate, up) || !MetalDenseOps.sameMatrices(gate, gradient)) {
            return MetalSwiGluDispatch.backward(graph, gradient, gate, up);
        }
        GpuBuffer gateGradient = graph.newOutputBuffer(gate.matrixRows(), gate.matrixCols());
        GpuBuffer upGradient = graph.newOutputBuffer(up.matrixRows(), up.matrixCols());
        graph.recordSwiGluBackward(gpuIn(gradient), gpuIn(gate), gpuIn(up),
                gateGradient, upGradient);
        return new SwiGluBackwardResult(gpuOut(gateGradient), gpuOut(upGradient));
    }

    static void validateScatterAddRowsInputs(Tensor target, int[] indices, Tensor grad) {
        if (indices.length != grad.matrixRows()) {
            throw new IllegalArgumentException(
                    "scatterAddRows: indices length " + indices.length + " must match grad rows " + grad.matrixRows());
        }
        if (target.matrixCols() != grad.matrixCols()) {
            throw new IllegalArgumentException(
                    "scatterAddRows: target cols " + target.matrixCols()
                            + " must match grad cols " + grad.matrixCols());
        }
    }

    void recordGpuScatterAddRowsAtomic(Tensor target, Tensor grad, Tensor indexTensor, int[] indices) {
        if (MetalHostZeroInput.isUnboundZero(target)) {
            graph.bindTensorToBuffer(target, graph.newZeroBuffer(target.matrixRows(), target.matrixCols()));
        }
        GpuBuffer gTarget = gpuIn(target);
        GpuBuffer gGrad = gpuIn(grad);
        GpuBuffer gIndices = gpuIn(indexTensor);

        graph.recordScatterAddRowsAtomic(gTarget, gIndices, gGrad,
                target.matrixRows(), target.matrixCols(), indices.length);
        gTarget.cpuStale = true;
        gTarget.needsUpload = false;
    }

    @Override
    public void scatterAddRows(Tensor target, int[] indices, Tensor grad) {
        validateScatterAddRowsInputs(target, indices, grad);
        if (!MetalDenseOps.matrix(target) || !MetalDenseOps.matrix(grad) || !MetalIndexMetadata.fitsFloat(indices)) {
            graph.layoutIndexAdd(target, 0, indices, grad);
            return;
        }
        if (indices.length == 0) return;

        Tensor indexTensor = immutableIntColumn(indices);

        recordGpuScatterAddRowsAtomic(target, grad, indexTensor, indices);
    }

    @Override
    public void releaseTemporaryResources() {
        graph.releaseTemporary();
    }

    @Override
    public void releaseResources() {
        graph.releaseAll();
    }
}
