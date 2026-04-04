package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;

import java.util.Random;

public final class MetalBackend implements TensorBackend {

    private final CpuBackend cpuFallback = new CpuBackend();
    private final ComputeGraph graph = new ComputeGraph(new MetalGpuRuntime());


    // ── Lazy helpers ──────────────────────────────────────────────

    /** Ensure input tensor has a GpuBuffer; upload from CPU if needed. */
    private GpuBuffer gpuIn(Tensor t) { return graph.ensureGpuBuffer(t); }

    /** Wrap a GPU output buffer in a tracked Tensor. */
    private Tensor gpuOut(GpuBuffer buf) {
        return graph.createOutputTensor(buf);
    }

    /** Force-materialize a tensor to CPU before a CPU-only op. */
    private void ensureCpu(Tensor t) {
        if (t.getGpuTag() instanceof GpuBuffer gb && gb.cpuStale) {
            graph.materialize(t);
        }
    }

    /** Materialize multiple tensors before a CPU-only op. */
    private void ensureCpu(Tensor... tensors) {
        for (Tensor t : tensors) ensureCpu(t);
    }

    // ── materializeTensor (called by Tensor.materialize()) ──────

    @Override
    public void materializeTensor(Tensor t) {
        graph.materialize(t);
    }

    // ── factories ──────────────────────────────────────────────────

    @Override public Tensor zeros(int rows, int cols) { return cpuFallback.zeros(rows, cols); }
    @Override public Tensor ones(int rows, int cols) { return cpuFallback.ones(rows, cols); }
    @Override public Tensor random(int rows, int cols, Random rand) { return cpuFallback.random(rows, cols, rand); }
    @Override public Tensor causalMask(int size) { return cpuFallback.causalMask(size); }
    @Override public Tensor unflattenToTensor(double[] flat, int rows, int cols) { return cpuFallback.unflattenToTensor(flat, rows, cols); }
    @Override public double[] flattenTensor(Tensor t) { ensureCpu(t); return cpuFallback.flattenTensor(t); }

    // ── LAZY matmul ────────────────────────────────────────────────

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        if (a.cols != b.rows) throw new IllegalArgumentException(
                "Shape mismatch for matmul: " + a.rows + "x" + a.cols + " vs " + b.rows + "x" + b.cols);
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, b.cols);
        graph.recordMatmul(ga, gb, gOut, a.rows, b.cols, a.cols);
        return gpuOut(gOut);
    }

    // ── LAZY element-wise binary ───────────────────────────────────

    @Override
    public Tensor add(Tensor a, Tensor b) {
        Tensor.requireSameShape(a, b, "add");
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordBinary(ComputeGraph.OP_ADD, ga, gb, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor subtract(Tensor a, Tensor b) {
        Tensor.requireSameShape(a, b, "subtract");
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordBinary(ComputeGraph.OP_SUBTRACT, ga, gb, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor multiply(Tensor a, Tensor b) {
        Tensor.requireSameShape(a, b, "multiply");
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordBinary(ComputeGraph.OP_MULTIPLY, ga, gb, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor divide(Tensor a, Tensor b) {
        Tensor.requireSameShape(a, b, "divide");
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordBinary(ComputeGraph.OP_DIVIDE, ga, gb, gOut);
        return gpuOut(gOut);
    }

    // ── broadcast (CPU only — no GPU kernel) ──────────────────────

    @Override public Tensor addRowVector(Tensor a, Tensor v) { ensureCpu(a, v); return cpuFallback.addRowVector(a, v); }
    @Override public Tensor addBroadcastCols(Tensor a, Tensor v) { ensureCpu(a, v); return cpuFallback.addBroadcastCols(a, v); }
    @Override public Tensor divideBroadcastCols(Tensor a, Tensor v) { ensureCpu(a, v); return cpuFallback.divideBroadcastCols(a, v); }
    @Override public Tensor subtractBroadcastCols(Tensor a, Tensor v) { ensureCpu(a, v); return cpuFallback.subtractBroadcastCols(a, v); }
    @Override public Tensor multiplyBroadcastCols(Tensor a, Tensor v) { ensureCpu(a, v); return cpuFallback.multiplyBroadcastCols(a, v); }
    @Override public Tensor addBroadcastRows(Tensor a, Tensor v) { ensureCpu(a, v); return cpuFallback.addBroadcastRows(a, v); }
    @Override public Tensor multiplyBroadcastRows(Tensor a, Tensor v) { ensureCpu(a, v); return cpuFallback.multiplyBroadcastRows(a, v); }

    // ── LAZY scalar ops ────────────────────────────────────────────

    @Override
    public Tensor multiplyScalar(Tensor a, double scalar) {
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordMultiplyScalar(ga, gOut, (float) scalar);
        return gpuOut(gOut);
    }

    @Override public Tensor addScalar(Tensor a, double scalar) { ensureCpu(a); return cpuFallback.addScalar(a, scalar); }
    @Override public Tensor divideScalar(Tensor a, double scalar) { ensureCpu(a); return cpuFallback.divideScalar(a, scalar); }

    // ── reductions (CPU only — need materialization) ───────────────

    @Override public Tensor sumRows(Tensor a) { ensureCpu(a); return cpuFallback.sumRows(a); }
    @Override public Tensor sumAlongRows(Tensor a) { ensureCpu(a); return cpuFallback.sumAlongRows(a); }
    @Override public Tensor sumAlongCols(Tensor a) { ensureCpu(a); return cpuFallback.sumAlongCols(a); }
    @Override public Tensor meanAlongRows(Tensor a) { ensureCpu(a); return cpuFallback.meanAlongRows(a); }
    @Override public Tensor varianceAlongRows(Tensor a) { ensureCpu(a); return cpuFallback.varianceAlongRows(a); }
    @Override public Tensor maxAlongRows(Tensor a) { ensureCpu(a); return cpuFallback.maxAlongRows(a); }
    @Override public double sum(Tensor a) { ensureCpu(a); return cpuFallback.sum(a); }
    @Override public double sumAbs(Tensor a) { ensureCpu(a); return cpuFallback.sumAbs(a); }

    // ── unary math — CPU only ──────────────────────────────────────

    @Override public Tensor clamp(Tensor a, double min, double max) { ensureCpu(a); return cpuFallback.clamp(a, min, max); }
    @Override public Tensor transpose(Tensor a) { ensureCpu(a); return cpuFallback.transpose(a); }
    @Override public Tensor pow(Tensor a, double exponent) { ensureCpu(a); return cpuFallback.pow(a, exponent); }

    // ── LAZY unary math ────────────────────────────────────────────

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

    // ── LAZY activations ───────────────────────────────────────────

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

    // ── LAZY softmax ───────────────────────────────────────────────

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

    // ── fused ops ──────────────────────────────────────────────────

    @Override public double crossEntropyLoss(Tensor logits, int[] targets) { ensureCpu(logits); return cpuFallback.crossEntropyLoss(logits, targets); }

    @Override
    public Tensor crossEntropyGradient(Tensor logits, int[] targets) {
        Tensor.requireTargetsMatchRows(logits, targets);
        Tensor probs = softmaxRows(logits);
        Tensor oneHot = new Tensor(logits.rows, logits.cols);
        for (int r = 0; r < logits.rows; r++) {
            oneHot.data[r][targets[r]] = 1.0;
        }
        Tensor grad = subtract(probs, oneHot);
        return multiplyScalar(grad, 1.0 / logits.rows);
    }

    @Override
    public void adamWUpdate(Tensor w, Tensor g, Tensor mt, Tensor vt,
                            double lr, double beta1, double beta2, double eps,
                            double weightDecay, double bc1, double bc2) {
        Tensor.requireSameShape(w, g, "adamWUpdate");
        Tensor.requireSameShape(w, mt, "adamWUpdate");
        Tensor.requireSameShape(w, vt, "adamWUpdate");

        GpuBuffer gw = gpuIn(w);
        GpuBuffer gg = gpuIn(g);
        GpuBuffer gmt = gpuIn(mt);
        GpuBuffer gvt = gpuIn(vt);
        graph.recordAdamWUpdate(
                gw, gg, gmt, gvt,
                (float) lr, (float) beta1, (float) beta2, (float) eps,
                (float) weightDecay, (float) bc1, (float) bc2,
                w.rows * w.cols
        );

        gw.cpuStale  = true;  gw.needsUpload  = false;
        gmt.cpuStale = true;  gmt.needsUpload = false;
        gvt.cpuStale = true;  gvt.needsUpload = false;
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

    // ── data accessors (CPU only) ──────────────────────────────────

    @Override public double get(Tensor t, int r, int c) { ensureCpu(t); return cpuFallback.get(t, r, c); }
    @Override public void set(Tensor t, int r, int c, double value) { ensureCpu(t); cpuFallback.set(t, r, c, value); }
    @Override public Tensor getRow(Tensor t, int row) { ensureCpu(t); return cpuFallback.getRow(t, row); }
    @Override public void setRow(Tensor t, int row, Tensor source, int srcRow) { ensureCpu(t, source); cpuFallback.setRow(t, row, source, srcRow); }
    @Override public Tensor sliceRows(Tensor t, int[] rowIndices, int cols) { ensureCpu(t); return cpuFallback.sliceRows(t, rowIndices, cols); }
    @Override public void scatterAddRows(Tensor target, int[] indices, Tensor grad) { ensureCpu(target, grad); cpuFallback.scatterAddRows(target, indices, grad); }
    @Override public Tensor sampleRows(Tensor t, int n, Random rnd) { ensureCpu(t); return cpuFallback.sampleRows(t, n, rnd); }

    // ── debug ──────────────────────────────────────────────────────

    @Override public void print(Tensor t, String label) { ensureCpu(t); cpuFallback.print(t, label); }

    @Override
    public void releaseResources() {
        graph.releaseAll();
    }
}