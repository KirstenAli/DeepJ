package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;

import java.util.List;

abstract class MetalTrainingOps extends MetalInPlaceOps {
    @Override
    public Tensor crossEntropyGradient(Tensor logits, int[] targets) {
        Tensor.requireTargetsMatchRows(logits, targets);
        Tensor targetTensor = immutableIntColumn(targets);
        GpuBuffer gLogits = gpuIn(logits);
        GpuBuffer gTargets = gpuIn(targetTensor);
        GpuBuffer gOut = graph.newOutputBuffer(logits.rows, logits.cols);
        graph.recordCrossEntropyGradient(gLogits, gTargets, gOut, logits.rows, logits.cols);
        return gpuOut(gOut);
    }

    @Override
    public CrossEntropyResult crossEntropy(Tensor logits, int[] targets) {
        Tensor.requireTargetsMatchRows(logits, targets);
        Tensor targetTensor = immutableIntColumn(targets);
        GpuBuffer losses = graph.newOutputBuffer(logits.rows, 1);
        GpuBuffer gradient = graph.newOutputBuffer(logits.rows, logits.cols);
        graph.recordCrossEntropy(gpuIn(logits), gpuIn(targetTensor), losses,
                gradient, logits.rows, logits.cols);
        Tensor meanLoss = divideScalar(sumRows(gpuOut(losses)), logits.rows);
        return new CrossEntropyResult(meanLoss, gpuOut(gradient));
    }

    @Override
    public float crossEntropyLoss(Tensor logits, int[] targets) {
        Tensor.requireTargetsMatchRows(logits, targets);

        Tensor targetTensor = immutableIntColumn(targets);
        GpuBuffer gLogits = gpuIn(logits);
        GpuBuffer gTargets = gpuIn(targetTensor);
        GpuBuffer gRowLosses = graph.newOutputBuffer(logits.rows, 1);
        graph.recordCrossEntropyLoss(gLogits, gTargets, gRowLosses, logits.rows, logits.cols);

        Tensor rowLosses = gpuOut(gRowLosses);
        Tensor scalar = sumRows(rowLosses).divideScalar(logits.rows);
        scalar.materialize();
        return scalar.data[0];
    }

    @Override
    public float crossEntropyLoss(Tensor logits, int[] targets, boolean[] mask) {
        Tensor losses = crossEntropyRowLosses(logits, targets);
        Tensor masked = multiply(losses, maskTensor(mask));
        Tensor scalar = divideScalar(sumRows(masked), includedRows(mask));
        scalar.materialize();
        return scalar.data[0];
    }

    @Override
    public Tensor crossEntropyGradient(Tensor logits, int[] targets, boolean[] mask) {
        Tensor gradient = crossEntropyGradient(logits, targets);
        Tensor masked = multiplyBroadcastCols(gradient, maskTensor(mask));
        return multiplyScalar(masked, (float) logits.rows / includedRows(mask));
    }

    Tensor crossEntropyRowLosses(Tensor logits, int[] targets) {
        Tensor targetTensor = immutableIntColumn(targets);
        GpuBuffer output = graph.newOutputBuffer(logits.rows, 1);
        graph.recordCrossEntropyLoss(gpuIn(logits), gpuIn(targetTensor), output,
                logits.rows, logits.cols);
        return gpuOut(output);
    }

    static Tensor maskTensor(boolean[] mask) {
        Tensor tensor = new Tensor(mask.length, 1);
        for (int row = 0; row < mask.length; row++) {
            tensor.data[row] = mask[row] ? 1.0f : 0.0f;
        }
        return tensor;
    }

    static int includedRows(boolean[] mask) {
        int count = 0;
        for (boolean included : mask) {
            if (included) count++;
        }
        return count;
    }

    @Override
    public void adamWUpdate(Tensor w, Tensor g, Tensor mt, Tensor vt,
                            float lr, float beta1, float beta2, float eps,
                            float weightDecay, float bc1, float bc2) {
        Tensor.requireSameShape(w, g, "adamWUpdate");
        Tensor.requireSameShape(w, mt, "adamWUpdate");
        Tensor.requireSameShape(w, vt, "adamWUpdate");

        GpuBuffer gw = gpuIn(w);
        GpuBuffer gg = gpuIn(g);
        GpuBuffer gmt = gpuIn(mt);
        GpuBuffer gvt = gpuIn(vt);
        graph.recordAdamWUpdate(
                gw, gg, gmt, gvt,
                lr, beta1, beta2, eps,
                weightDecay, bc1, bc2,
                w.rows * w.cols
        );

        markUpdated(gw);
        markUpdated(gmt);
        markUpdated(gvt);
    }

    static void markUpdated(GpuBuffer buffer) {
        buffer.cpuStale = true;
        buffer.needsUpload = false;
    }

    @Override
    public Tensor layerNormBackward(Tensor dXHat, Tensor xHat, Tensor std, int dim) {
        Tensor.requireSameShape(dXHat, xHat, "layerNormBackward");
        if (dim != dXHat.cols) throw new IllegalArgumentException(
                "layerNormBackward: dim=" + dim + " must equal tensor cols=" + dXHat.cols);
        if (std.rows != dXHat.rows || std.cols != 1) throw new IllegalArgumentException(
                "layerNormBackward: std must be " + dXHat.rows + "x1 but got " + std.rows + "x" + std.cols);

        GpuBuffer gDXHat = gpuIn(dXHat);
        GpuBuffer gXHat  = gpuIn(xHat);
        GpuBuffer gStd   = gpuIn(std);
        GpuBuffer gOut   = graph.newOutputBuffer(dXHat.rows, dXHat.cols);
        graph.recordLayerNormBackward(gDXHat, gXHat, gStd, gOut, dXHat.rows, dXHat.cols);
        return gpuOut(gOut);
    }

    @Override
    public RmsNormResult rmsNorm(Tensor input, Tensor gamma, float epsilon) {
        requireRmsNormShapes(input, gamma, input.rows, 1);
        GpuBuffer output = graph.newOutputBuffer(input.rows, input.cols);
        GpuBuffer normalized = graph.newOutputBuffer(input.rows, input.cols);
        GpuBuffer rms = graph.newOutputBuffer(input.rows, 1);
        graph.recordRmsNorm(gpuIn(input), gpuIn(gamma), output, normalized,
                rms, input.rows, input.cols, epsilon);
        return new RmsNormResult(gpuOut(output), gpuOut(normalized), gpuOut(rms));
    }

    @Override
    public Tensor rmsNormBackward(Tensor gradient, Tensor normalized,
                                  Tensor rms, Tensor gamma) {
        Tensor.requireSameShape(gradient, normalized, "rmsNormBackward");
        requireRmsNormShapes(gradient, gamma, rms.rows, rms.cols);
        GpuBuffer output = graph.newOutputBuffer(gradient.rows, gradient.cols);
        graph.recordRmsNormBackward(gpuIn(gradient), gpuIn(normalized), gpuIn(rms),
                gpuIn(gamma), output, gradient.rows, gradient.cols);
        return gpuOut(output);
    }

    static void requireRmsNormShapes(Tensor input, Tensor gamma,
                                             int rmsRows, int rmsCols) {
        if (gamma.rows != 1 || gamma.cols != input.cols) {
            throw new IllegalArgumentException("RMSNorm gamma must match input columns");
        }
        if (rmsRows != input.rows || rmsCols != 1) {
            throw new IllegalArgumentException("RMSNorm scale must match input rows");
        }
    }

    @Override
    public Tensor swiGlu(Tensor gate, Tensor up) {
        Tensor.requireSameShape(gate, up, "swiGlu");
        GpuBuffer fused = graph.newOutputBuffer(gate.rows, gate.cols);
        graph.recordSwiGlu(gpuIn(gate), gpuIn(up), fused);
        return gpuOut(fused);
    }

    @Override
    public SwiGluBackwardResult swiGluBackward(Tensor gradient, Tensor gate, Tensor up) {
        Tensor.requireSameShape(gradient, gate, "swiGluBackward");
        Tensor.requireSameShape(gate, up, "swiGluBackward");
        GpuBuffer gateGradient = graph.newOutputBuffer(gate.rows, gate.cols);
        GpuBuffer upGradient = graph.newOutputBuffer(up.rows, up.cols);
        graph.recordSwiGluBackward(gpuIn(gradient), gpuIn(gate), gpuIn(up),
                gateGradient, upGradient);
        return new SwiGluBackwardResult(gpuOut(gateGradient), gpuOut(upGradient));
    }

    static void validateScatterAddRowsInputs(Tensor target, int[] indices, Tensor grad) {
        if (indices.length != grad.rows) {
            throw new IllegalArgumentException(
                    "scatterAddRows: indices length " + indices.length + " must match grad rows " + grad.rows);
        }
        if (target.cols != grad.cols) {
            throw new IllegalArgumentException(
                    "scatterAddRows: target cols " + target.cols + " must match grad cols " + grad.cols);
        }
    }

    void recordGpuScatterAddRowsAtomic(Tensor target, Tensor grad, Tensor indexTensor, int[] indices) {
        GpuBuffer gTarget = gpuIn(target);
        GpuBuffer gGrad = gpuIn(grad);
        GpuBuffer gIndices = gpuIn(indexTensor);

        graph.recordScatterAddRowsAtomic(gTarget, gIndices, gGrad, target.rows, target.cols, indices.length);
        gTarget.cpuStale = true;
        gTarget.needsUpload = false;
    }

    @Override
    public void scatterAddRows(Tensor target, int[] indices, Tensor grad) {
        validateScatterAddRowsInputs(target, indices, grad);
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
