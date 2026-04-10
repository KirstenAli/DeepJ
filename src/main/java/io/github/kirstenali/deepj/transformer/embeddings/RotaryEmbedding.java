package io.github.kirstenali.deepj.transformer.embeddings;

import io.github.kirstenali.deepj.tensor.Tensor;

/**
 * Rotary Positional Embedding (RoPE) — used in Llama, Mistral, Qwen, DeepSeek, and GPT-NeoX.
 *
 * <p>Unlike additive positional embeddings, RoPE has <em>no learnable parameters</em>.
 * It encodes position by rotating pairs of Q and K head dimensions by position-dependent angles,
 * which causes relative-position information to appear naturally in dot-product attention scores.
 *
 * <p>Applied <em>inside</em> {@link io.github.kirstenali.deepj.layers.transformer.MultiHeadSelfAttention}
 * after the Q/K projections are split into heads, before the scaled dot-product is computed.
 *
 * <h3>Math (per position {@code t}, pair index {@code i}):</h3>
 * <pre>
 *   θ_{t,i}  = t / 10000^(2i / headDim)
 *
 *   x_rot[2i]   =  x[2i]  · cos θ  −  x[2i+1] · sin θ
 *   x_rot[2i+1] =  x[2i]  · sin θ  +  x[2i+1] · cos θ
 * </pre>
 *
 * <h3>Backward (transpose rotation — negate sin):</h3>
 * <pre>
 *   dx[2i]   =  d[2i]  · cos θ  +  d[2i+1] · sin θ
 *   dx[2i+1] = −d[2i]  · sin θ  +  d[2i+1] · cos θ
 * </pre>
 *
 * <p><b>Input tensor shape convention (split-head layout):</b>
 * {@code [nHeads × seqLen, headDim]} — head {@code h} occupies rows
 * {@code [h·seqLen .. (h+1)·seqLen)}, position {@code t} is at row {@code h·seqLen + t}.
 */
public final class RotaryEmbedding {

    private final int halfDim;
    private final float[][] cosTable;  // [maxSeqLen × halfDim]
    private final float[][] sinTable;  // [maxSeqLen × halfDim]

    /**
     * Pre-computes the cos/sin rotation tables.
     *
     * @param headDim   dimension of each attention head ({@code dModel / nHeads}); must be even
     * @param maxSeqLen maximum sequence length to support
     */
    public RotaryEmbedding(int headDim, int maxSeqLen) {
        if (headDim <= 0 || headDim % 2 != 0) {
            throw new IllegalArgumentException("headDim must be a positive even number, got " + headDim);
        }
        if (maxSeqLen <= 0) {
            throw new IllegalArgumentException("maxSeqLen must be > 0");
        }

        this.halfDim  = headDim / 2;
        this.cosTable = new float[maxSeqLen][this.halfDim];
        this.sinTable = new float[maxSeqLen][this.halfDim];

        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int i = 0; i < this.halfDim; i++) {
                double theta = pos / Math.pow(10_000.0, (2.0 * i) / headDim);
                cosTable[pos][i] = (float) Math.cos(theta);
                sinTable[pos][i] = (float) Math.sin(theta);
            }
        }
    }

    /**
     * Apply rotary embeddings to a split-head tensor (forward direction).
     *
     * @param t       split-head tensor, shape {@code [nHeads·seqLen × headDim]}
     * @param seqLen  number of positions
     * @param nHeads  number of attention heads
     * @return rotated tensor with the same shape
     */
    public Tensor apply(Tensor t, int seqLen, int nHeads) {
        validateSeqLen(seqLen);
        t.materialize();
        Tensor result = Tensor.zeros(t.rows, t.cols);
        for (int h = 0; h < nHeads; h++)
            for (int pos = 0; pos < seqLen; pos++)
                rotatePositionForward(result, t, h * seqLen + pos, pos);
        return result;
    }

    /**
     * Apply the transpose (inverse) rotation — used in the backward pass.
     *
     * @param t       gradient tensor, same shape as the forward input
     * @param seqLen  number of positions
     * @param nHeads  number of attention heads
     * @return un-rotated gradient with the same shape
     */
    public Tensor applyBackward(Tensor t, int seqLen, int nHeads) {
        validateSeqLen(seqLen);
        t.materialize();
        Tensor result = Tensor.zeros(t.rows, t.cols);
        for (int h = 0; h < nHeads; h++)
            for (int pos = 0; pos < seqLen; pos++)
                rotatePositionInverse(result, t, h * seqLen + pos, pos);
        return result;
    }

    // ── internal ────────────────────────────────────────────────────────────

    /** Forward rotation for every dimension pair in one position row. */
    private void rotatePositionForward(Tensor result, Tensor t, int row, int pos) {
        int base = row * t.cols;
        for (int i = 0; i < halfDim; i++) {
            float cos = cosTable[pos][i];
            float sin = sinTable[pos][i];
            float x0  = t.data[base + 2 * i];
            float x1  = t.data[base + 2 * i + 1];
            result.data[base + 2 * i]     = x0 * cos - x1 * sin;
            result.data[base + 2 * i + 1] = x0 * sin + x1 * cos;
        }
    }

    /** Inverse (transpose) rotation for every dimension pair in one position row. */
    private void rotatePositionInverse(Tensor result, Tensor t, int row, int pos) {
        int base = row * t.cols;
        for (int i = 0; i < halfDim; i++) {
            float cos = cosTable[pos][i];
            float sin = sinTable[pos][i];
            float x0  = t.data[base + 2 * i];
            float x1  = t.data[base + 2 * i + 1];
            result.data[base + 2 * i]     = x0 * cos + x1 * sin;
            result.data[base + 2 * i + 1] = -x0 * sin + x1 * cos;
        }
    }

    private void validateSeqLen(int seqLen) {
        if (seqLen > cosTable.length) {
            throw new IllegalArgumentException(
                    "seqLen " + seqLen + " exceeds maxSeqLen " + cosTable.length);
        }
    }
}
