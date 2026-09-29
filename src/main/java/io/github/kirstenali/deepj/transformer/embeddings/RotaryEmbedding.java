package io.github.kirstenali.deepj.transformer.embeddings;

import io.github.kirstenali.deepj.tensor.Tensor;

public final class RotaryEmbedding {

    private final int headDim;
    private final int halfDim;
    private final Tensor cosTable;
    private final Tensor sinTable;

    public RotaryEmbedding(int headDim, int maxSeqLen) {
        validateDimensions(headDim, maxSeqLen);
        this.headDim = headDim;
        this.halfDim = headDim / 2;
        this.cosTable = Tensor.zeros(maxSeqLen, halfDim).retainDeviceBuffer();
        this.sinTable = Tensor.zeros(maxSeqLen, halfDim).retainDeviceBuffer();
        populateTables(maxSeqLen);
    }

    private static void validateDimensions(int headDim, int maxSeqLen) {
        if (headDim <= 0 || headDim % 2 != 0) {
            throw new IllegalArgumentException("headDim must be a positive even number, got " + headDim);
        }
        if (maxSeqLen <= 0) {
            throw new IllegalArgumentException("maxSeqLen must be > 0");
        }
    }

    private void populateTables(int maxSeqLen) {
        for (int position = 0; position < maxSeqLen; position++) {
            for (int dimension = 0; dimension < halfDim; dimension++) {
                float theta = (float) (position
                        / Math.pow(10_000.0f, (2.0f * dimension) / headDim));
                int index = position * halfDim + dimension;
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
