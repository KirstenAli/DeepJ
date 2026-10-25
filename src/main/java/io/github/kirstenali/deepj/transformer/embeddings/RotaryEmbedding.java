package io.github.kirstenali.deepj.transformer.embeddings;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.autograd.LayoutOps;
import io.github.kirstenali.deepj.autograd.GradMode;

import java.util.stream.IntStream;

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
        return LayoutOps.rotary(t, cosTable, sinTable, seqLen, false);
    }

    public Tensor apply(Tensor tensor, int length, int heads, int offset) {
        if (offset == 0) return apply(tensor, length, heads);
        validateInput(tensor, length, heads);
        if (offset < 0 || (long) offset + length > cosTable.rows) {
            throw new IllegalArgumentException("Rotary position exceeds the configured context");
        }
        int[] rows = IntStream.range(offset, offset + length).toArray();
        return LayoutOps.rotary(tensor, LayoutOps.gather(cosTable, rows),
                LayoutOps.gather(sinTable, rows), length, false);
    }

    public Tensor applyBackward(Tensor t, int seqLen, int nHeads) {
        validateInput(t, seqLen, nHeads);
        return LayoutOps.rotary(t, cosTable, sinTable, seqLen, true);
    }

    /** Continuing positions for rolling inference; persistent rotary tables never grow. */
    public Tensor applyInference(Tensor input, int length, int heads, int offset) {
        if ((long) offset + length <= cosTable.rows) return apply(input, length, heads, offset);
        if (GradMode.isEnabled()) throw new IllegalStateException("Rolling RoPE requires noGrad inference");
        validateInput(input, length, heads);
        Math.addExact(offset, length);
        RotaryPositionRows rows = RotaryPositionRows.at(headDim, length, offset);
        return LayoutOps.rotary(input, rows.cosine(), rows.sine(), length, false);
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
