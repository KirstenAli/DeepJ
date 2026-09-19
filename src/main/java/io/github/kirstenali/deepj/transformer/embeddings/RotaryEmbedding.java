package io.github.kirstenali.deepj.transformer.embeddings;

import io.github.kirstenali.deepj.tensor.Tensor;

public final class RotaryEmbedding {

    private final int headDim;
    private final int halfDim;
    private final float[][] cosTable;
    private final float[][] sinTable;

    public RotaryEmbedding(int headDim, int maxSeqLen) {
        if (headDim <= 0 || headDim % 2 != 0) {
            throw new IllegalArgumentException("headDim must be a positive even number, got " + headDim);
        }
        if (maxSeqLen <= 0) {
            throw new IllegalArgumentException("maxSeqLen must be > 0");
        }

        this.headDim = headDim;
        this.halfDim  = headDim / 2;
        this.cosTable = new float[maxSeqLen][this.halfDim];
        this.sinTable = new float[maxSeqLen][this.halfDim];

        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int i = 0; i < this.halfDim; i++) {
                float theta = (float) (pos / Math.pow(10_000.0f, (2.0f * i) / headDim));
                cosTable[pos][i] = (float) Math.cos(theta);
                sinTable[pos][i] = (float) Math.sin(theta);
            }
        }
    }

    public Tensor apply(Tensor t, int seqLen, int nHeads) {
        validateInput(t, seqLen, nHeads);
        t.materialize();
        Tensor result = Tensor.zeros(t.rows, t.cols);
        for (int h = 0; h < nHeads; h++)
            for (int pos = 0; pos < seqLen; pos++)
                rotatePositionForward(result, t, h * seqLen + pos, pos);
        return result;
    }

    public Tensor applyBackward(Tensor t, int seqLen, int nHeads) {
        validateInput(t, seqLen, nHeads);
        t.materialize();
        Tensor result = Tensor.zeros(t.rows, t.cols);
        for (int h = 0; h < nHeads; h++)
            for (int pos = 0; pos < seqLen; pos++)
                rotatePositionInverse(result, t, h * seqLen + pos, pos);
        return result;
    }

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

    public int headDim() {
        return headDim;
    }

    private void validateInput(Tensor tensor, int seqLen, int nHeads) {
        if (seqLen <= 0 || seqLen > cosTable.length) {
            throw new IllegalArgumentException(
                    "seqLen must be in [1, " + cosTable.length + "], got " + seqLen);
        }
        if (nHeads <= 0) throw new IllegalArgumentException("nHeads must be > 0");
        if (tensor.cols != headDim) throw new IllegalArgumentException("Tensor width must equal headDim");
        if ((long) tensor.rows != (long) seqLen * nHeads) {
            throw new IllegalArgumentException("Tensor rows must equal seqLen * nHeads");
        }
    }
}
