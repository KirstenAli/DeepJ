package io.github.kirstenali.deepj.transformer.embeddings;

import io.github.kirstenali.deepj.tensor.Tensor;

public final class RotaryEmbedding {

    private final int headDim;
    private final int halfDim;
    private final Tensor cosTable;
    private final Tensor sinTable;

    public RotaryEmbedding(int headDim, int maxSeqLen) {
        if (headDim <= 0 || headDim % 2 != 0) {
            throw new IllegalArgumentException("headDim must be a positive even number, got " + headDim);
        }
        if (maxSeqLen <= 0) {
            throw new IllegalArgumentException("maxSeqLen must be > 0");
        }

        this.headDim = headDim;
        this.halfDim  = headDim / 2;
        this.cosTable = Tensor.zeros(maxSeqLen, halfDim).retainDeviceBuffer();
        this.sinTable = Tensor.zeros(maxSeqLen, halfDim).retainDeviceBuffer();

        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int i = 0; i < this.halfDim; i++) {
                float theta = (float) (pos / Math.pow(10_000.0f, (2.0f * i) / headDim));
                int index = pos * halfDim + i;
                cosTable.data[index] = (float) Math.cos(theta);
                sinTable.data[index] = (float) Math.sin(theta);
            }
        }
    }

    public Tensor apply(Tensor t, int seqLen, int nHeads) {
        validateInput(t, seqLen, nHeads);
        return Tensor.backend().rotary(t, cosTable, sinTable, seqLen, false);
    }

    public Tensor applyBackward(Tensor t, int seqLen, int nHeads) {
        validateInput(t, seqLen, nHeads);
        return Tensor.backend().rotary(t, cosTable, sinTable, seqLen, true);
    }

    public int headDim() {
        return headDim;
    }

    private void validateInput(Tensor tensor, int seqLen, int nHeads) {
        if (seqLen <= 0 || seqLen > cosTable.rows) {
            throw new IllegalArgumentException(
                    "seqLen must be in [1, " + cosTable.rows + "], got " + seqLen);
        }
        if (nHeads <= 0) throw new IllegalArgumentException("nHeads must be > 0");
        if (tensor.cols != headDim) throw new IllegalArgumentException("Tensor width must equal headDim");
        if ((long) tensor.rows != (long) seqLen * nHeads) {
            throw new IllegalArgumentException("Tensor rows must equal seqLen * nHeads");
        }
    }
}
