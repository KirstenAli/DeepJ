package io.github.kirstenali.deepj.tensor;

/** Common validation for the existing rank-two CPU and native attention kernels. */
public final class MatrixShapeChecks {
    private MatrixShapeChecks() {}

    public static void requireMatrix(Tensor input) {
        if (input.rank() != 2) throw new IllegalArgumentException("This matrix helper requires rank two");
    }

    public static void requireRowMaximum(Tensor input) {
        requireMatrix(input);
        if (input.size(1) == 0) throw new IllegalArgumentException("Maximum cannot reduce an empty axis");
    }

    public static void requireCausal(Tensor input, int sequenceLength) {
        if (sequenceLength <= 0 || input.matrixCols() != sequenceLength
                || input.matrixRows() % sequenceLength != 0) {
            throw new IllegalArgumentException("Scores must contain complete square attention heads");
        }
    }

    public static void requireRotary(Tensor input, Tensor cosine, Tensor sine, int sequenceLength) {
        if (sequenceLength <= 0 || input.matrixRows() % sequenceLength != 0 || input.matrixCols() % 2 != 0) {
            throw new IllegalArgumentException("Invalid rotary input shape");
        }
        if (cosine.matrixRows() < sequenceLength || cosine.matrixCols() != input.matrixCols() / 2
                || sine.matrixRows() != cosine.matrixRows() || sine.matrixCols() != cosine.matrixCols()) {
            throw new IllegalArgumentException("Invalid rotary table shape");
        }
    }
}
