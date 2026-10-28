package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;

final class TensorBackendDefaults {

    private TensorBackendDefaults() {}

    static Tensor splitHeads(Tensor input, int heads) {
        requireHeads(input.matrixCols(), heads);
        input.materialize();
        int headDim = input.matrixCols() / heads;
        Tensor output = new Tensor(heads * input.matrixRows(), headDim);
        DeepJExecutor.forRange(0, input.matrixRows(), row -> splitRow(input, output, row, heads, headDim));
        return output;
    }

    static Tensor sliceRows(Tensor input, int[] rows) {
        input.materialize();
        Tensor output = new Tensor(rows.length, input.matrixCols());
        DeepJExecutor.forRange(0, rows.length, index -> copyRow(input, output, rows[index], index));
        return output;
    }

    private static void copyRow(Tensor input, Tensor output, int sourceRow, int targetRow) {
        if (sourceRow < 0 || sourceRow >= input.matrixRows()) {
            throw new IllegalArgumentException("Row index out of range: " + sourceRow);
        }
        System.arraycopy(input.rawData(), sourceRow * input.matrixCols(),
                output.rawData(), targetRow * output.matrixCols(), input.matrixCols());
    }

    private static void splitRow(Tensor input, Tensor output, int row, int heads, int headDim) {
        for (int head = 0; head < heads; head++) {
            int source = row * input.matrixCols() + head * headDim;
            int target = (head * input.matrixRows() + row) * headDim;
            System.arraycopy(input.rawData(), source, output.rawData(), target, headDim);
        }
    }

    static Tensor mergeHeads(Tensor input, int heads) {
        requireHeads(input.matrixRows(), heads);
        input.materialize();
        int rows = input.matrixRows() / heads;
        Tensor output = new Tensor(rows, heads * input.matrixCols());
        DeepJExecutor.forRange(0, rows, row -> mergeRow(input, output, row, heads));
        return output;
    }

    private static void mergeRow(Tensor input, Tensor output, int row, int heads) {
        for (int head = 0; head < heads; head++) {
            int source = (head * output.matrixRows() + row) * input.matrixCols();
            int target = row * output.matrixCols() + head * input.matrixCols();
            System.arraycopy(input.rawData(), source, output.rawData(), target, input.matrixCols());
        }
    }

    static Tensor batchedMatmul(Tensor left, Tensor right, int batches,
                                boolean transposeLeft, boolean transposeRight) {
        PackedMatmulShape shape = PackedMatmulShape.of(left, right, batches, transposeLeft, transposeRight);
        return CpuPackedMatmul.apply(left, right, batches, shape);
    }

    static Tensor causalMask(Tensor input, int sequenceLength) {
        MatrixShapeChecks.requireCausal(input, sequenceLength);
        if (!CpuNdElementwise.matrix(input)) return CpuNdPosition.mask(input, sequenceLength);
        input.materialize();
        Tensor output = new Tensor(input.matrixRows(), input.matrixCols());
        DeepJExecutor.forRange(0, input.matrixRows(), row -> maskRow(input, output, row, sequenceLength));
        return output;
    }

    private static void maskRow(Tensor input, Tensor output, int row, int sequenceLength) {
        int base = row * input.matrixCols();
        int position = row % sequenceLength;
        for (int col = 0; col < input.matrixCols(); col++) {
            output.rawData()[base + col] = col > position ? -1e9f : input.rawData()[base + col];
        }
    }

    static Tensor rotary(Tensor input, Tensor cosine, Tensor sine,
                         int sequenceLength, boolean inverse) {
        MatrixShapeChecks.requireRotary(input, cosine, sine, sequenceLength);
        if (!CpuNdElementwise.matrix(input) || !CpuNdElementwise.matrix(cosine) || !CpuNdElementwise.matrix(sine)) {
            return CpuNdPosition.rotary(input, cosine, sine, sequenceLength, inverse);
        }
        input.materialize();
        cosine.materialize();
        sine.materialize();
        Tensor output = new Tensor(input.matrixRows(), input.matrixCols());
        DeepJExecutor.forRange(0, input.matrixRows(), row ->
                rotateRow(input, cosine, sine, output, row, sequenceLength, inverse));
        return output;
    }

    private static void rotateRow(Tensor input, Tensor cosine, Tensor sine, Tensor output,
                                  int row, int sequenceLength, boolean inverse) {
        int position = row % sequenceLength;
        for (int pair = 0; pair < input.matrixCols() / 2; pair++) {
            rotatePair(input, cosine, sine, output, row, position, pair, inverse);
        }
    }

    private static void rotatePair(Tensor input, Tensor cosine, Tensor sine, Tensor output,
                                   int row, int position, int pair, boolean inverse) {
        int inputBase = row * input.matrixCols() + pair * 2;
        int tableIndex = position * cosine.matrixCols() + pair;
        float direction = inverse ? -1.0f : 1.0f;
        float x = input.rawData()[inputBase], y = input.rawData()[inputBase + 1];
        float cos = cosine.rawData()[tableIndex], sin = sine.rawData()[tableIndex] * direction;
        output.rawData()[inputBase] = x * cos - y * sin;
        output.rawData()[inputBase + 1] = x * sin + y * cos;
    }

    private static void requireHeads(int dimension, int heads) {
        if (heads <= 0 || dimension % heads != 0) {
            throw new IllegalArgumentException("Dimension must be divisible by a positive head count");
        }
    }

}
