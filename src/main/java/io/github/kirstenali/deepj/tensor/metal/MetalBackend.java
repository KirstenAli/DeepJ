package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;

import java.util.HashMap;
import java.util.Map;

public final class MetalBackend implements TensorBackend {
    private final ComputeGraph graph = new ComputeGraph(new MetalGpuRuntime());
    private final Map<Integer, Tensor> intColumnScratchByLength = new HashMap<>();

    private static void requireRowVector(Tensor rowVector, Tensor target) {
        if (rowVector.rows != 1 || rowVector.cols != target.cols) {
            throw new IllegalArgumentException(
                    "Expected row vector 1x" + target.cols + " but got " + rowVector.rows + "x" + rowVector.cols);
        }
    }

    private static void requireColVector(Tensor colVector, Tensor target) {
        if (colVector.rows != target.rows || colVector.cols != 1) {
            throw new IllegalArgumentException(
                    "Expected col vector " + target.rows + "x1 but got " + colVector.rows + "x" + colVector.cols);
        }
    }

    // ── Lazy helpers ──────────────────────────────────────────────

    /** Ensure input tensor has a GpuBuffer; upload from CPU if needed. */
    private GpuBuffer gpuIn(Tensor t) { return graph.ensureGpuBuffer(t); }

    /** Wrap a GPU output buffer in a tracked Tensor. */
    private Tensor gpuOut(GpuBuffer buf) {
        return graph.createOutputTensor(buf);
    }

    private static void markNeedsUpload(Tensor t) {
        if (t.getGpuTag() instanceof GpuBuffer gb) {
            gb.needsUpload = true;
            gb.cpuStale = false;
        }
    }

    private Tensor reusableIntColumn(int[] values) {
        Tensor t = intColumnScratchByLength.get(values.length);
        if (t == null) {
            t = new Tensor(values.length, 1);
            intColumnScratchByLength.put(values.length, t);
        }
        for (int i = 0; i < values.length; i++) {
            t.data[i] = values[i];
        }
        markNeedsUpload(t);
        return t;
    }

    // ── materializeTensor (called by Tensor.materialize()) ──────

    @Override
    public void materializeTensor(Tensor t) {
        graph.materialize(t);
    }

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

    // ── LAZY broadcast ──────────────────────────────────────────────

    @Override
    public Tensor addRowVector(Tensor a, Tensor v) {
        requireRowVector(v, a);
        GpuBuffer ga = gpuIn(a), gv = gpuIn(v);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordRowBroadcast(ComputeGraph.OP_ADD_ROW_VECTOR, ga, gv, gOut, a.rows, a.cols);
        return gpuOut(gOut);
    }

    @Override
    public Tensor addBroadcastCols(Tensor a, Tensor v) {
        requireColVector(v, a);
        GpuBuffer ga = gpuIn(a), gv = gpuIn(v);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordColBroadcast(ComputeGraph.OP_ADD_BROADCAST_COLS, ga, gv, gOut, a.rows, a.cols);
        return gpuOut(gOut);
    }

    @Override
    public Tensor divideBroadcastCols(Tensor a, Tensor v) {
        requireColVector(v, a);
        GpuBuffer ga = gpuIn(a), gv = gpuIn(v);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordColBroadcast(ComputeGraph.OP_DIVIDE_BROADCAST_COLS, ga, gv, gOut, a.rows, a.cols);
        return gpuOut(gOut);
    }

    @Override
    public Tensor subtractBroadcastCols(Tensor a, Tensor v) {
        requireColVector(v, a);
        GpuBuffer ga = gpuIn(a), gv = gpuIn(v);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordColBroadcast(ComputeGraph.OP_SUBTRACT_BROADCAST_COLS, ga, gv, gOut, a.rows, a.cols);
        return gpuOut(gOut);
    }

    @Override
    public Tensor multiplyBroadcastCols(Tensor a, Tensor v) {
        requireColVector(v, a);
        GpuBuffer ga = gpuIn(a), gv = gpuIn(v);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordColBroadcast(ComputeGraph.OP_MULTIPLY_BROADCAST_COLS, ga, gv, gOut, a.rows, a.cols);
        return gpuOut(gOut);
    }

    @Override
    public Tensor addBroadcastRows(Tensor a, Tensor v) {
        return addRowVector(a, v);
    }

    @Override
    public Tensor multiplyBroadcastRows(Tensor a, Tensor v) {
        requireRowVector(v, a);
        GpuBuffer ga = gpuIn(a), gv = gpuIn(v);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordRowBroadcast(ComputeGraph.OP_MULTIPLY_BROADCAST_ROWS, ga, gv, gOut, a.rows, a.cols);
        return gpuOut(gOut);
    }

    // ── LAZY scalar ops ────────────────────────────────────────────

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

    // ── LAZY reductions ─────────────────────────────────────────────

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

    // ── unary math ─────────────────────────────────────────────────

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

    // ── LAZY in-place ops ─────────────────────────────────────────

    private void bindInPlaceResult(Tensor target, GpuBuffer out) {
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

    // ── fused ops ──────────────────────────────────────────────────

    @Override
    public Tensor crossEntropyGradient(Tensor logits, int[] targets) {
        Tensor.requireTargetsMatchRows(logits, targets);
        Tensor targetTensor = reusableIntColumn(targets);
        GpuBuffer gLogits = gpuIn(logits);
        GpuBuffer gTargets = gpuIn(targetTensor);
        GpuBuffer gOut = graph.newOutputBuffer(logits.rows, logits.cols);
        graph.recordCrossEntropyGradient(gLogits, gTargets, gOut, logits.rows, logits.cols);
        return gpuOut(gOut);
    }

    @Override
    public float crossEntropyLoss(Tensor logits, int[] targets) {
        Tensor.requireTargetsMatchRows(logits, targets);

        Tensor targetTensor = reusableIntColumn(targets);
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

    private static void validateScatterAddRowsInputs(Tensor target, int[] indices, Tensor grad) {
        if (indices.length != grad.rows) {
            throw new IllegalArgumentException(
                    "scatterAddRows: indices length " + indices.length + " must match grad rows " + grad.rows);
        }
        if (target.cols != grad.cols) {
            throw new IllegalArgumentException(
                    "scatterAddRows: target cols " + target.cols + " must match grad cols " + grad.cols);
        }
    }

    private void recordGpuScatterAddRowsAtomic(Tensor target, Tensor grad, Tensor indexTensor, int[] indices) {
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

        Tensor indexTensor = reusableIntColumn(indices);
        // Always use the atomic GPU path so duplicate indices stay parallel.
        recordGpuScatterAddRowsAtomic(target, grad, indexTensor, indices);
    }

    @Override
    public void releaseResources() {
        graph.releaseAll();
        intColumnScratchByLength.clear();
    }
}