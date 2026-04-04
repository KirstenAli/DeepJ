package io.github.kirstenali.deepj.tensor;

import java.util.Random;

public interface TensorBackend {

    // ── factories ──────────────────────────────────────────────────────
    Tensor zeros(int rows, int cols);
    Tensor ones(int rows, int cols);
    Tensor random(int rows, int cols, Random rand);
    Tensor causalMask(int size);

    Tensor unflattenToTensor(double[] flat, int rows, int cols);
    double[] flattenTensor(Tensor t);

    // ── element-wise binary ────────────────────────────────────────────
    Tensor matmul(Tensor a, Tensor b);

    Tensor add(Tensor a, Tensor b);
    Tensor subtract(Tensor a, Tensor b);
    Tensor multiply(Tensor a, Tensor b);
    Tensor divide(Tensor a, Tensor b);

    // ── broadcast ──────────────────────────────────────────────────────
    Tensor addRowVector(Tensor a, Tensor rowVector);

    Tensor addBroadcastCols(Tensor a, Tensor colVector);
    Tensor divideBroadcastCols(Tensor a, Tensor colVector);
    Tensor subtractBroadcastCols(Tensor a, Tensor colVector);
    Tensor multiplyBroadcastCols(Tensor a, Tensor colVector);

    Tensor addBroadcastRows(Tensor a, Tensor rowVector);
    Tensor multiplyBroadcastRows(Tensor a, Tensor rowVector);

    // ── scalar ops ─────────────────────────────────────────────────────
    Tensor multiplyScalar(Tensor a, double scalar);
    Tensor addScalar(Tensor a, double scalar);
    Tensor divideScalar(Tensor a, double scalar);

    // ── reductions ─────────────────────────────────────────────────────
    Tensor sumRows(Tensor a);
    Tensor sumAlongRows(Tensor a);
    Tensor sumAlongCols(Tensor a);
    Tensor meanAlongRows(Tensor a);
    Tensor varianceAlongRows(Tensor a);
    Tensor maxAlongRows(Tensor a);

    double sum(Tensor a);
    double sumAbs(Tensor a);

    // ── unary math ─────────────────────────────────────────────────────
    Tensor clamp(Tensor a, double min, double max);
    Tensor transpose(Tensor a);
    Tensor sqrt(Tensor a);
    Tensor pow(Tensor a, double exponent);
    Tensor neg(Tensor a);
    Tensor exp(Tensor a);
    Tensor log(Tensor a);

    // ── activation element-wise ────────────────────────────────────────
    Tensor tanh(Tensor a);
    Tensor sigmoid(Tensor a);
    Tensor relu(Tensor a);
    Tensor reluBackward(Tensor input, Tensor gradOutput);
    Tensor gelu(Tensor a);
    Tensor geluBackward(Tensor input, Tensor gradOutput);

    // ── row-wise compound ──────────────────────────────────────────────
    Tensor softmaxRows(Tensor logits);
    Tensor softmaxBackward(Tensor gradOutput, Tensor softmaxOut);

    // ── fused high-level ops ───────────────────────────────────────────
    double crossEntropyLoss(Tensor logits, int[] targets);
    Tensor crossEntropyGradient(Tensor logits, int[] targets);

    /**
     * In-place AdamW update.  Mutates w, mt, vt.
     */
    void adamWUpdate(Tensor w, Tensor g, Tensor mt, Tensor vt,
                     double lr, double beta1, double beta2, double eps,
                     double weightDecay, double bc1, double bc2);

    /**
     * LayerNorm backward through normalization (given dXHat, xHat, std).
     */
    Tensor layerNormBackward(Tensor dXHat, Tensor xHat, Tensor std, int dim);

    // ── data accessors (for code that must touch elements) ─────────────
    double get(Tensor t, int r, int c);
    void set(Tensor t, int r, int c, double value);
    Tensor getRow(Tensor t, int row);
    void setRow(Tensor t, int row, Tensor source, int srcRow);

    /** Gather rows by index (for embedding lookup). */
    Tensor sliceRows(Tensor t, int[] rowIndices, int cols);

    /** Scatter-add: target.data[indices[i]] += grad.data[i] (for embedding backward). */
    void scatterAddRows(Tensor target, int[] indices, Tensor grad);

    /** Sample random rows from t. */
    Tensor sampleRows(Tensor t, int n, Random rnd);

    // ── debug ──────────────────────────────────────────────────────────
    void print(Tensor t, String label);

    // ── in-place operations (write result back into first argument) ─
    //
    // Default implementations allocate a temporary tensor and copy.
    // CpuBackend overrides these for true zero-allocation in-place.

    default void addInPlace(Tensor a, Tensor b)              { copyInto(add(a, b), a); }
    default void subtractInPlace(Tensor a, Tensor b)          { copyInto(subtract(a, b), a); }
    default void multiplyInPlace(Tensor a, Tensor b)          { copyInto(multiply(a, b), a); }
    default void divideInPlace(Tensor a, Tensor b)            { copyInto(divide(a, b), a); }

    default void multiplyScalarInPlace(Tensor a, double s)    { copyInto(multiplyScalar(a, s), a); }
    default void addScalarInPlace(Tensor a, double s)         { copyInto(addScalar(a, s), a); }
    default void divideScalarInPlace(Tensor a, double s)      { copyInto(divideScalar(a, s), a); }

    default void sqrtInPlace(Tensor a)     { copyInto(sqrt(a), a); }
    default void negInPlace(Tensor a)      { copyInto(neg(a), a); }
    default void expInPlace(Tensor a)      { copyInto(exp(a), a); }
    default void logInPlace(Tensor a)      { copyInto(log(a), a); }
    default void reluInPlace(Tensor a)     { copyInto(relu(a), a); }
    default void geluInPlace(Tensor a)     { copyInto(gelu(a), a); }
    default void tanhInPlace(Tensor a)     { copyInto(tanh(a), a); }
    default void sigmoidInPlace(Tensor a)  { copyInto(sigmoid(a), a); }

    /** Copy src data into dst (same shape). Used by default in-place implementations. */
    private static void copyInto(Tensor src, Tensor dst) {
        for (int r = 0; r < src.rows; r++)
            System.arraycopy(src.data[r], 0, dst.data[r], 0, src.cols);
    }

    // ── lazy execution support ─────────────────────────────────────────
    /**
     * Materialize a tensor: flush any pending GPU computation and download
     * the result to the tensor's CPU data[][]. Default is a no-op (for CpuBackend).
     */
    default void materializeTensor(Tensor t) { /* no-op for eager backends */ }

    /**
     * Release backend-owned resources (GPU buffers, native handles, etc.).
     * Default is a no-op for backends without external resources.
     */
    default void releaseResources() { /* no-op for eager backends */ }
}