package io.github.kirstenali.deepj.tensor;

public interface TensorBackend {



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
    Tensor multiplyScalar(Tensor a, float scalar);
    Tensor addScalar(Tensor a, float scalar);
    Tensor divideScalar(Tensor a, float scalar);

    // ── reductions ─────────────────────────────────────────────────────
    Tensor sumRows(Tensor a);
    Tensor sumAlongRows(Tensor a);
    Tensor sumAlongCols(Tensor a);
    Tensor meanAlongRows(Tensor a);
    Tensor varianceAlongRows(Tensor a);

    // ── unary math ─────────────────────────────────────────────────────
    Tensor transpose(Tensor a);
    Tensor sqrt(Tensor a);
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
    Tensor crossEntropyGradient(Tensor logits, int[] targets);

    /**
     * In-place AdamW update.  Mutates w, mt, vt.
     */
    void adamWUpdate(Tensor w, Tensor g, Tensor mt, Tensor vt,
                     float lr, float beta1, float beta2, float eps,
                     float weightDecay, float bc1, float bc2);

    /**
     * LayerNorm backward through normalization (given dXHat, xHat, std).
     */
    Tensor layerNormBackward(Tensor dXHat, Tensor xHat, Tensor std, int dim);


    // ── in-place operations (write result back into first argument) ─
    //
    // Default implementations allocate a temporary tensor and copy.
    // CpuBackend overrides these for true zero-allocation in-place.

    default void addInPlace(Tensor a, Tensor b)              { copyIntoMaterialized(add(a, b), a); }
    default void subtractInPlace(Tensor a, Tensor b)          { copyIntoMaterialized(subtract(a, b), a); }
    default void multiplyInPlace(Tensor a, Tensor b)          { copyIntoMaterialized(multiply(a, b), a); }
    default void divideInPlace(Tensor a, Tensor b)            { copyIntoMaterialized(divide(a, b), a); }

    default void multiplyScalarInPlace(Tensor a, float s)    { copyIntoMaterialized(multiplyScalar(a, s), a); }
    default void addScalarInPlace(Tensor a, float s)         { copyIntoMaterialized(addScalar(a, s), a); }
    default void divideScalarInPlace(Tensor a, float s)      { copyIntoMaterialized(divideScalar(a, s), a); }

    default void sqrtInPlace(Tensor a)     { copyIntoMaterialized(sqrt(a), a); }
    default void negInPlace(Tensor a)      { copyIntoMaterialized(neg(a), a); }
    default void expInPlace(Tensor a)      { copyIntoMaterialized(exp(a), a); }
    default void logInPlace(Tensor a)      { copyIntoMaterialized(log(a), a); }
    default void reluInPlace(Tensor a)     { copyIntoMaterialized(relu(a), a); }
    default void geluInPlace(Tensor a)     { copyIntoMaterialized(gelu(a), a); }
    default void tanhInPlace(Tensor a)     { copyIntoMaterialized(tanh(a), a); }
    default void sigmoidInPlace(Tensor a)  { copyIntoMaterialized(sigmoid(a), a); }

    /** Copy src data into dst (same shape). Used by default in-place implementations. */
    private static void copyInto(Tensor src, Tensor dst) {
        System.arraycopy(src.data, 0, dst.data, 0, src.data.length);
    }

    /**
     * For lazy backends, force source/destination CPU views up to date before copy.
     */
    private void copyIntoMaterialized(Tensor src, Tensor dst) {
        materializeTensor(src);
        materializeTensor(dst);
        copyInto(src, dst);
    }

    // ── lazy execution support ─────────────────────────────────────────
    /**
     * Materialize a tensor: flush any pending GPU computation and download
     * the result to the tensor's CPU data[]. Default is a no-op (for CpuBackend).
     */
    default void materializeTensor(Tensor t) { /* no-op for eager backends */ }

    /**
     * Release backend-owned resources (GPU buffers, native handles, etc.).
     * Default is a no-op for backends without external resources.
     */
    default void releaseResources() { /* no-op for eager backends */ }
}