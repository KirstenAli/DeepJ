package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;

import java.util.Arrays;
import java.util.Random;

public class Tensor {
    /** Flat row-major storage: element (r, c) lives at data[r * cols + c]. */
    public final float[] data;
    public final int rows, cols;

    /**
     * GPU handle — set by a GPU-backed {@link TensorBackend} when this tensor is part of
     * a lazy {@link ComputeGraph}. Null for CPU-only tensors.
     */
    Object gpuTag;

    /** Get the GPU handle (used by GPU backends). */
    public Object getGpuTag() { return gpuTag; }
    /** Set the GPU handle (used by GPU backends). */
    public void setGpuTag(Object tag) { this.gpuTag = tag; }

    private static volatile TensorBackend BACKEND = new CpuBackend(); // default

    public static void setBackend(TensorBackend backend) {
        if (backend == null) throw new IllegalArgumentException("backend cannot be null");
        BACKEND = backend;
    }

    public static TensorBackend backend() {
        return BACKEND;
    }

    public Tensor(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new float[rows * cols];
    }

    /** Copy constructor — creates an independent deep copy. */
    public Tensor(Tensor source) {
        if (source == null) throw new IllegalArgumentException("source cannot be null");
        source.materialize();
        this.rows = source.rows;
        this.cols = source.cols;
        this.data = Arrays.copyOf(source.data, source.data.length);
        // Never share backend-owned GPU handles across tensor copies.
        this.gpuTag = null;
    }

    /**
     * Returns a fresh {@code float[]} copy of row {@code r}.
     * Convenience for tests and debugging; not a view.
     */
    public float[] rowData(int r) {
        return Arrays.copyOfRange(data, r * cols, (r + 1) * cols);
    }

    /**
     * Ensures this tensor's {@code data[]} is up to date with any pending GPU computation.
     * No-op if this tensor has no GPU handle or is already materialized.
     * Call this before reading {@code data[]} directly.
     */
    public void materialize() {
        if (gpuTag != null) {
            backend().materializeTensor(this);
        }
    }

    // ── instance ops (element-wise binary) ──────────────────────────
    public Tensor matmul(Tensor other) { return backend().matmul(this, other); }
    public Tensor add(Tensor other) { return backend().add(this, other); }
    public Tensor subtract(Tensor other) { return backend().subtract(this, other); }
    public Tensor multiply(Tensor other) { return backend().multiply(this, other); }
    public Tensor divide(Tensor other) { return backend().divide(this, other); }

    // ── broadcast ───────────────────────────────────────────────────
    public Tensor addRowVector(Tensor rowVector) { return backend().addRowVector(this, rowVector); }
    public Tensor addBroadcastCols(Tensor colVector) { return backend().addBroadcastCols(this, colVector); }
    public Tensor divideBroadcastCols(Tensor colVector) { return backend().divideBroadcastCols(this, colVector); }
    public Tensor subtractBroadcastCols(Tensor colVector) { return backend().subtractBroadcastCols(this, colVector); }
    public Tensor multiplyBroadcastCols(Tensor colVector) { return backend().multiplyBroadcastCols(this, colVector); }
    public Tensor addBroadcastRows(Tensor rowVector) { return backend().addBroadcastRows(this, rowVector); }
    public Tensor multiplyBroadcastRows(Tensor rowVector) { return backend().multiplyBroadcastRows(this, rowVector); }

    // ── scalar ops ──────────────────────────────────────────────────
    public Tensor multiplyScalar(float s) { return backend().multiplyScalar(this, s); }
    public Tensor addScalar(float s) { return backend().addScalar(this, s); }
    public Tensor divideScalar(float s) { return backend().divideScalar(this, s); }

    // ── reductions ──────────────────────────────────────────────────
    public Tensor sumRows() { return backend().sumRows(this); }
    public Tensor sumAlongRows() { return backend().sumAlongRows(this); }
    public Tensor sumAlongCols() { return backend().sumAlongCols(this); }
    public Tensor meanAlongRows() { return backend().meanAlongRows(this); }
    public Tensor varianceAlongRows() { return backend().varianceAlongRows(this); }
    public Tensor maxAlongRows() { return backend().maxAlongRows(this); }
    // ── reductions (scalar-returning — trigger materialization) ──
    public float sum() { materialize(); return backend().sum(this); }
    public float sumAbs() { materialize(); return backend().sumAbs(this); }

    // ── unary math ──────────────────────────────────────────────────
    public Tensor clamp(float min, float max) { return backend().clamp(this, min, max); }
    public Tensor transpose() { return backend().transpose(this); }
    public Tensor sqrt() { return backend().sqrt(this); }
    public Tensor pow(float exponent) { return backend().pow(this, exponent); }
    public Tensor neg() { return backend().neg(this); }
    public Tensor exp() { return backend().exp(this); }
    public Tensor log() { return backend().log(this); }

    // ── activations (element-wise) ──────────────────────────────────
    public Tensor tanhActivation() { return backend().tanh(this); }
    public Tensor sigmoidActivation() { return backend().sigmoid(this); }
    public Tensor reluActivation() { return backend().relu(this); }
    public Tensor reluBackward(Tensor gradOutput) { return backend().reluBackward(this, gradOutput); }
    public Tensor geluActivation() { return backend().gelu(this); }
    public Tensor geluBackward(Tensor gradOutput) { return backend().geluBackward(this, gradOutput); }

    // ── in-place ops (zero allocation, mutates this, returns this) ──
    public Tensor addInPlace(Tensor b)           { backend().addInPlace(this, b); return this; }
    public Tensor subtractInPlace(Tensor b)      { backend().subtractInPlace(this, b); return this; }
    public Tensor multiplyInPlace(Tensor b)      { backend().multiplyInPlace(this, b); return this; }
    public Tensor divideInPlace(Tensor b)        { backend().divideInPlace(this, b); return this; }

    public Tensor multiplyScalarInPlace(float s) { backend().multiplyScalarInPlace(this, s); return this; }
    public Tensor addScalarInPlace(float s)      { backend().addScalarInPlace(this, s); return this; }
    public Tensor divideScalarInPlace(float s)   { backend().divideScalarInPlace(this, s); return this; }

    public Tensor sqrtInPlace()    { backend().sqrtInPlace(this); return this; }
    public Tensor negInPlace()     { backend().negInPlace(this); return this; }
    public Tensor expInPlace()     { backend().expInPlace(this); return this; }
    public Tensor logInPlace()     { backend().logInPlace(this); return this; }
    public Tensor reluInPlace()    { backend().reluInPlace(this); return this; }
    public Tensor geluInPlace()    { backend().geluInPlace(this); return this; }
    public Tensor tanhInPlace()    { backend().tanhInPlace(this); return this; }
    public Tensor sigmoidInPlace() { backend().sigmoidInPlace(this); return this; }

    // ── row-wise compound ───────────────────────────────────────────
    public Tensor softmaxRows() { return backend().softmaxRows(this); }
    public Tensor softmaxBackward(Tensor softmaxOut) { return backend().softmaxBackward(this, softmaxOut); }

    // ── fused high-level ops ────────────────────────────────────────
    public float crossEntropyLoss(int[] targets) { materialize(); return backend().crossEntropyLoss(this, targets); }
    public Tensor crossEntropyGradient(int[] targets) { return backend().crossEntropyGradient(this, targets); }

    public static void adamWUpdate(Tensor w, Tensor g, Tensor mt, Tensor vt,
                                   float lr, float beta1, float beta2, float eps,
                                   float weightDecay, float bc1, float bc2) {
        backend().adamWUpdate(w, g, mt, vt, lr, beta1, beta2, eps, weightDecay, bc1, bc2);
    }

    public static Tensor layerNormBackward(Tensor dXHat, Tensor xHat, Tensor std, int dim) {
        return backend().layerNormBackward(dXHat, xHat, std, dim);
    }

    // ── data accessors (trigger materialization) ────────────────
    public float get(int r, int c) { materialize(); return backend().get(this, r, c); }
    public void set(int r, int c, float value) { materialize(); backend().set(this, r, c, value); }
    public Tensor getRow(int row) { materialize(); return backend().getRow(this, row); }
    public void setRow(int row, Tensor source, int srcRow) { materialize(); source.materialize(); backend().setRow(this, row, source, srcRow); }

    public static Tensor sliceRows(Tensor t, int[] rowIndices, int cols) {
        return backend().sliceRows(t, rowIndices, cols);
    }
    public static void scatterAddRows(Tensor target, int[] indices, Tensor grad) {
        backend().scatterAddRows(target, indices, grad);
    }
    public static Tensor sampleRows(Tensor t, int n, Random rnd) {
        return backend().sampleRows(t, n, rnd);
    }

    // ── debug (trigger materialization) ─────────────────────────
    public void print(String label) { materialize(); backend().print(this, label); }

    // ── static factories ────────────────────────────────────────────
    public static Tensor zeros(int rows, int cols) { return backend().zeros(rows, cols); }
    public static Tensor ones(int rows, int cols) { return backend().ones(rows, cols); }
    public static Tensor random(int rows, int cols, Random rand) { return backend().random(rows, cols, rand); }
    public static Tensor causalMask(int size) { return backend().causalMask(size); }

    /**
     * Build a tensor from 2-D row-major data.
     * Preferred API for literal matrix construction.
     */
    public static Tensor from2D(float[][] data) {
        int rows = data.length;
        int cols = data[0].length;
        Tensor t = new Tensor(rows, cols);
        for (int r = 0; r < rows; r++) {
            if (data[r].length != cols) {
                throw new IllegalArgumentException("All rows must have the same length (expected " + cols + ")");
            }
            System.arraycopy(data[r], 0, t.data, r * cols, cols);
        }
        return t;
    }

    // ── shape checks ────────────────────────────────────────────────
    public static void requireSameShape(Tensor a, Tensor b, String op) {
        if (a.rows != b.rows || a.cols != b.cols) {
            throw new IllegalArgumentException(
                    "Shape mismatch for " + op + ": " + a.rows + "x" + a.cols +
                            " vs " + b.rows + "x" + b.cols);
        }
    }

    public static void requireTargetsMatchRows(Tensor logits, int[] targets) {
        if (targets.length != logits.rows) {
            throw new IllegalArgumentException(
                    "targets length " + targets.length + " must match logits rows " + logits.rows);
        }
    }
}