package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;

import java.util.Random;

public final class MetalBackend implements TensorBackend {

    private final CpuBackend cpuFallback = new CpuBackend();
    private final ComputeGraph graph = new ComputeGraph(new MetalGpuRuntime());

    private long matmulGpuThreshold;
    private long elementwiseGpuThreshold;
    private boolean logDispatches;

    /**
     * Create a MetalBackend with default thresholds.
     * Matmul: 1,048,576 (m*k*n work units).
     * Element-wise: 4,096 total elements.
     */
    public MetalBackend() {
        this(128L * 128 * 64, 4096, false);
    }

    /**
     * Create a MetalBackend with custom thresholds.
     *
     * @param matmulGpuThreshold      minimum m*k*n work for matmul GPU dispatch
     * @param elementwiseGpuThreshold minimum element count for element-wise GPU dispatch
     * @param logDispatches           if true, print CPU/GPU dispatch decisions to stderr
     */
    public MetalBackend(long matmulGpuThreshold, long elementwiseGpuThreshold, boolean logDispatches) {
        this.matmulGpuThreshold = matmulGpuThreshold;
        this.elementwiseGpuThreshold = elementwiseGpuThreshold;
        this.logDispatches = logDispatches;
    }

    /** Returns true if the Metal native library was loaded successfully. */
    public static boolean isGpuAvailable() { return MetalNative.AVAILABLE; }

    public void setMatmulGpuThreshold(long threshold)      { this.matmulGpuThreshold = threshold; }
    public void setElementwiseGpuThreshold(long threshold)  { this.elementwiseGpuThreshold = threshold; }
    public void setLogDispatches(boolean log)               { this.logDispatches = log; }


    // ── Threshold checks ──────────────────────────────────────────

    private boolean useGpu(long work, String op, int rows, int cols) {
        boolean gpu = MetalNative.AVAILABLE && work >= elementwiseGpuThreshold;
        if (logDispatches) {
            System.err.printf("[Metal] %s [%dx%d] n=%,d → %s%n", op, rows, cols, work, gpu ? "GPU" : "CPU");
        }
        return gpu;
    }

    private boolean useGpuMatmul(Tensor a, Tensor b) {
        long work = (long) a.rows * a.cols * b.cols;
        boolean gpu = MetalNative.AVAILABLE && work >= matmulGpuThreshold;
        if (logDispatches) {
            System.err.printf("[Metal] matmul [%dx%d]·[%dx%d] work=%,d → %s%n",
                    a.rows, a.cols, b.rows, b.cols, work, gpu ? "GPU" : "CPU");
        }
        return gpu;
    }

    private boolean useGpuEW(Tensor a, String op) {
        return useGpu((long) a.rows * a.cols, op, a.rows, a.cols);
    }

    // ── Lazy helpers ──────────────────────────────────────────────

    /** Ensure input tensor has a GpuBuffer; upload from CPU if needed. */
    private GpuBuffer gpuIn(Tensor t) { return graph.ensureGpuBuffer(t); }

    /** Allocate a GPU output buffer and create a result Tensor. */
    private Tensor gpuOut(int rows, int cols, GpuBuffer buf) {
        return graph.createOutputTensor(buf);
    }

    /** Force-materialize a tensor to CPU before a CPU-fallback op. */
    private void ensureCpu(Tensor t) {
        if (t.getGpuTag() instanceof GpuBuffer gb && gb.cpuStale) {
            graph.materialize(t);
        }
    }

    /** Materialize multiple tensors before CPU fallback. */
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

    // ── LAZY element-wise binary ───────────────────────────────────

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        if (a.cols != b.rows) throw new IllegalArgumentException(
                "Shape mismatch for matmul: " + a.rows + "x" + a.cols + " vs " + b.rows + "x" + b.cols);
        if (!useGpuMatmul(a, b)) { ensureCpu(a, b); return cpuFallback.matmul(a, b); }

        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, b.cols);
        graph.recordMatmul(ga, gb, gOut, a.rows, b.cols, a.cols);
        return gpuOut(a.rows, b.cols, gOut);
    }

    @Override
    public Tensor add(Tensor a, Tensor b) {
        Tensor.requireSameShape(a, b, "add");
        if (!useGpuEW(a, "add")) { ensureCpu(a, b); return cpuFallback.add(a, b); }

        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordBinary(ComputeGraph.OP_ADD, ga, gb, gOut);
        return gpuOut(a.rows, a.cols, gOut);
    }

    @Override
    public Tensor subtract(Tensor a, Tensor b) {
        Tensor.requireSameShape(a, b, "subtract");
        if (!useGpuEW(a, "subtract")) { ensureCpu(a, b); return cpuFallback.subtract(a, b); }

        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordBinary(ComputeGraph.OP_SUBTRACT, ga, gb, gOut);
        return gpuOut(a.rows, a.cols, gOut);
    }

    @Override
    public Tensor multiply(Tensor a, Tensor b) {
        Tensor.requireSameShape(a, b, "multiply");
        if (!useGpuEW(a, "multiply")) { ensureCpu(a, b); return cpuFallback.multiply(a, b); }

        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordBinary(ComputeGraph.OP_MULTIPLY, ga, gb, gOut);
        return gpuOut(a.rows, a.cols, gOut);
    }

    @Override
    public Tensor divide(Tensor a, Tensor b) {
        Tensor.requireSameShape(a, b, "divide");
        if (!useGpuEW(a, "divide")) { ensureCpu(a, b); return cpuFallback.divide(a, b); }

        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordBinary(ComputeGraph.OP_DIVIDE, ga, gb, gOut);
        return gpuOut(a.rows, a.cols, gOut);
    }

    // ── broadcast (CPU fallback — small tensors) ───────────────────

    @Override public Tensor addRowVector(Tensor a, Tensor v) { ensureCpu(a, v); return cpuFallback.addRowVector(a, v); }
    @Override public Tensor addBroadcastCols(Tensor a, Tensor v) { ensureCpu(a, v); return cpuFallback.addBroadcastCols(a, v); }
    @Override public Tensor divideBroadcastCols(Tensor a, Tensor v) { ensureCpu(a, v); return cpuFallback.divideBroadcastCols(a, v); }
    @Override public Tensor subtractBroadcastCols(Tensor a, Tensor v) { ensureCpu(a, v); return cpuFallback.subtractBroadcastCols(a, v); }
    @Override public Tensor multiplyBroadcastCols(Tensor a, Tensor v) { ensureCpu(a, v); return cpuFallback.multiplyBroadcastCols(a, v); }
    @Override public Tensor addBroadcastRows(Tensor a, Tensor v) { ensureCpu(a, v); return cpuFallback.addBroadcastRows(a, v); }
    @Override public Tensor multiplyBroadcastRows(Tensor a, Tensor v) { ensureCpu(a, v); return cpuFallback.multiplyBroadcastRows(a, v); }

    // ── scalar ops ─────────────────────────────────────────────────

    @Override
    public Tensor multiplyScalar(Tensor a, double scalar) {
        if (!useGpuEW(a, "multiplyScalar")) { ensureCpu(a); return cpuFallback.multiplyScalar(a, scalar); }

        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordMultiplyScalar(ga, gOut, (float) scalar);
        return gpuOut(a.rows, a.cols, gOut);
    }

    @Override public Tensor addScalar(Tensor a, double scalar) { ensureCpu(a); return cpuFallback.addScalar(a, scalar); }
    @Override public Tensor divideScalar(Tensor a, double scalar) { ensureCpu(a); return cpuFallback.divideScalar(a, scalar); }

    // ── reductions (CPU fallback — need materialization) ───────────

    @Override public Tensor sumRows(Tensor a) { ensureCpu(a); return cpuFallback.sumRows(a); }
    @Override public Tensor sumAlongRows(Tensor a) { ensureCpu(a); return cpuFallback.sumAlongRows(a); }
    @Override public Tensor sumAlongCols(Tensor a) { ensureCpu(a); return cpuFallback.sumAlongCols(a); }
    @Override public Tensor meanAlongRows(Tensor a) { ensureCpu(a); return cpuFallback.meanAlongRows(a); }
    @Override public Tensor varianceAlongRows(Tensor a) { ensureCpu(a); return cpuFallback.varianceAlongRows(a); }
    @Override public Tensor maxAlongRows(Tensor a) { ensureCpu(a); return cpuFallback.maxAlongRows(a); }
    @Override public double sum(Tensor a) { ensureCpu(a); return cpuFallback.sum(a); }
    @Override public double sumAbs(Tensor a) { ensureCpu(a); return cpuFallback.sumAbs(a); }

    // ── unary math ─────────────────────────────────────────────────

    @Override public Tensor clamp(Tensor a, double min, double max) { ensureCpu(a); return cpuFallback.clamp(a, min, max); }
    @Override public Tensor transpose(Tensor a) { ensureCpu(a); return cpuFallback.transpose(a); }
    @Override public Tensor pow(Tensor a, double exponent) { ensureCpu(a); return cpuFallback.pow(a, exponent); }

    @Override
    public Tensor sqrt(Tensor a) {
        if (!useGpuEW(a, "sqrt")) { ensureCpu(a); return cpuFallback.sqrt(a); }
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_SQRT, ga, gOut);
        return gpuOut(a.rows, a.cols, gOut);
    }

    @Override
    public Tensor neg(Tensor a) {
        if (!useGpuEW(a, "neg")) { ensureCpu(a); return cpuFallback.neg(a); }
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_NEG, ga, gOut);
        return gpuOut(a.rows, a.cols, gOut);
    }

    @Override
    public Tensor exp(Tensor a) {
        if (!useGpuEW(a, "exp")) { ensureCpu(a); return cpuFallback.exp(a); }
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_EXP, ga, gOut);
        return gpuOut(a.rows, a.cols, gOut);
    }

    @Override
    public Tensor log(Tensor a) {
        if (!useGpuEW(a, "log")) { ensureCpu(a); return cpuFallback.log(a); }
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_LOG, ga, gOut);
        return gpuOut(a.rows, a.cols, gOut);
    }

    // ── LAZY activations ───────────────────────────────────────────

    @Override
    public Tensor tanh(Tensor a) {
        if (!useGpuEW(a, "tanh")) { ensureCpu(a); return cpuFallback.tanh(a); }
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_TANH, ga, gOut);
        return gpuOut(a.rows, a.cols, gOut);
    }

    @Override
    public Tensor sigmoid(Tensor a) {
        if (!useGpuEW(a, "sigmoid")) { ensureCpu(a); return cpuFallback.sigmoid(a); }
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_SIGMOID, ga, gOut);
        return gpuOut(a.rows, a.cols, gOut);
    }

    @Override
    public Tensor relu(Tensor a) {
        if (!useGpuEW(a, "relu")) { ensureCpu(a); return cpuFallback.relu(a); }
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_RELU, ga, gOut);
        return gpuOut(a.rows, a.cols, gOut);
    }

    @Override
    public Tensor reluBackward(Tensor input, Tensor gradOutput) {
        if (!useGpuEW(input, "reluBackward")) { ensureCpu(input, gradOutput); return cpuFallback.reluBackward(input, gradOutput); }
        GpuBuffer gi = gpuIn(input), gg = gpuIn(gradOutput);
        GpuBuffer gOut = graph.newOutputBuffer(input.rows, input.cols);
        graph.recordBinary(ComputeGraph.OP_RELU_BACKWARD, gi, gg, gOut);
        return gpuOut(input.rows, input.cols, gOut);
    }

    @Override
    public Tensor gelu(Tensor a) {
        if (!useGpuEW(a, "gelu")) { ensureCpu(a); return cpuFallback.gelu(a); }
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordUnary(ComputeGraph.OP_GELU, ga, gOut);
        return gpuOut(a.rows, a.cols, gOut);
    }

    @Override
    public Tensor geluBackward(Tensor input, Tensor gradOutput) {
        if (!useGpuEW(input, "geluBackward")) { ensureCpu(input, gradOutput); return cpuFallback.geluBackward(input, gradOutput); }
        GpuBuffer gi = gpuIn(input), gg = gpuIn(gradOutput);
        GpuBuffer gOut = graph.newOutputBuffer(input.rows, input.cols);
        graph.recordBinary(ComputeGraph.OP_GELU_BACKWARD, gi, gg, gOut);
        return gpuOut(input.rows, input.cols, gOut);
    }

    // ── LAZY softmax ───────────────────────────────────────────────

    @Override
    public Tensor softmaxRows(Tensor logits) {
        if (!useGpuEW(logits, "softmaxRows")) { ensureCpu(logits); return cpuFallback.softmaxRows(logits); }
        GpuBuffer ga = gpuIn(logits);
        GpuBuffer gOut = graph.newOutputBuffer(logits.rows, logits.cols);
        graph.recordSoftmaxRows(ga, gOut, logits.rows, logits.cols);
        return gpuOut(logits.rows, logits.cols, gOut);
    }

    @Override
    public Tensor softmaxBackward(Tensor gradOutput, Tensor softmaxOut) {
        Tensor.requireSameShape(gradOutput, softmaxOut, "softmaxBackward");
        if (!useGpuEW(gradOutput, "softmaxBackward")) {
            ensureCpu(gradOutput, softmaxOut);
            return cpuFallback.softmaxBackward(gradOutput, softmaxOut);
        }

        GpuBuffer gGrad = gpuIn(gradOutput);
        GpuBuffer gSoftmax = gpuIn(softmaxOut);
        GpuBuffer gOut = graph.newOutputBuffer(gradOutput.rows, gradOutput.cols);
        graph.recordSoftmaxBackward(gGrad, gSoftmax, gOut, gradOutput.rows, gradOutput.cols);
        return gpuOut(gradOutput.rows, gradOutput.cols, gOut);
    }

    // ── fused (CPU fallback — materialization needed) ──────────────

    @Override public double crossEntropyLoss(Tensor logits, int[] targets) { ensureCpu(logits); return cpuFallback.crossEntropyLoss(logits, targets); }
    @Override public Tensor crossEntropyGradient(Tensor logits, int[] targets) { ensureCpu(logits); return cpuFallback.crossEntropyGradient(logits, targets); }

    @Override
    public void adamWUpdate(Tensor w, Tensor g, Tensor mt, Tensor vt,
                            double lr, double beta1, double beta2, double eps,
                            double weightDecay, double bc1, double bc2) {
        ensureCpu(w, g, mt, vt);
        cpuFallback.adamWUpdate(w, g, mt, vt, lr, beta1, beta2, eps, weightDecay, bc1, bc2);
        // After CPU update, invalidate any stale GPU buffer for w
        if (w.getGpuTag() instanceof GpuBuffer gb) { gb.needsUpload = true; gb.cpuStale = false; }
        if (mt.getGpuTag() instanceof GpuBuffer gb) { gb.needsUpload = true; gb.cpuStale = false; }
        if (vt.getGpuTag() instanceof GpuBuffer gb) { gb.needsUpload = true; gb.cpuStale = false; }
    }

    @Override
    public Tensor layerNormBackward(Tensor dXHat, Tensor xHat, Tensor std, int dim) {
        Tensor.requireSameShape(dXHat, xHat, "layerNormBackward");
        if (dim != dXHat.cols) {
            ensureCpu(dXHat, xHat, std);
            return cpuFallback.layerNormBackward(dXHat, xHat, std, dim);
        }
        if (std.rows != dXHat.rows || std.cols != 1) {
            throw new IllegalArgumentException("Shape mismatch for layerNormBackward std: expected "
                    + dXHat.rows + "x1 but got " + std.rows + "x" + std.cols);
        }
        if (!useGpuEW(dXHat, "layerNormBackward")) {
            ensureCpu(dXHat, xHat, std);
            return cpuFallback.layerNormBackward(dXHat, xHat, std, dim);
        }

        GpuBuffer gDXHat = gpuIn(dXHat);
        GpuBuffer gXHat = gpuIn(xHat);
        GpuBuffer gStd = gpuIn(std);
        GpuBuffer gOut = graph.newOutputBuffer(dXHat.rows, dXHat.cols);
        graph.recordLayerNormBackward(gDXHat, gXHat, gStd, gOut, dXHat.rows, dXHat.cols);
        return gpuOut(dXHat.rows, dXHat.cols, gOut);
    }

    // ── data accessors ─────────────────────────────────────────────

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