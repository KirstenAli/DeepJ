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
    Tensor maxAlongRows(Tensor a);
    float sum(Tensor a);
    float sumAbs(Tensor a);

    // ── unary math ─────────────────────────────────────────────────────
    Tensor transpose(Tensor a);
    Tensor clamp(Tensor a, float min, float max);
    Tensor sqrt(Tensor a);
    Tensor pow(Tensor a, float exponent);
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
    float crossEntropyLoss(Tensor logits, int[] targets);
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

    // ── row access/scatter helpers ───────────────────────────────────────
    void scatterAddRows(Tensor target, int[] indices, Tensor grad);


    // ── in-place operations (write result back into first argument) ─
    void addInPlace(Tensor a, Tensor b);
    void subtractInPlace(Tensor a, Tensor b);
    void multiplyInPlace(Tensor a, Tensor b);
    void divideInPlace(Tensor a, Tensor b);

    void multiplyScalarInPlace(Tensor a, float s);
    void addScalarInPlace(Tensor a, float s);
    void divideScalarInPlace(Tensor a, float s);

    void sqrtInPlace(Tensor a);
    void negInPlace(Tensor a);
    void expInPlace(Tensor a);
    void logInPlace(Tensor a);
    void reluInPlace(Tensor a);
    void geluInPlace(Tensor a);
    void tanhInPlace(Tensor a);
    void sigmoidInPlace(Tensor a);

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