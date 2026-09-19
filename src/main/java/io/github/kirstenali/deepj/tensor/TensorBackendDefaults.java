package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;

final class TensorBackendDefaults {

    private TensorBackendDefaults() {}

    static Tensor splitHeads(Tensor input, int heads) {
        requireHeads(input.cols, heads);
        input.materialize();
        int headDim = input.cols / heads;
        Tensor output = new Tensor(heads * input.rows, headDim);
        DeepJExecutor.forRange(0, input.rows, row -> splitRow(input, output, row, heads, headDim));
        return output;
    }

    static Tensor sliceRows(Tensor input, int[] rows) {
        input.materialize();
        Tensor output = new Tensor(rows.length, input.cols);
        DeepJExecutor.forRange(0, rows.length, index -> copyRow(input, output, rows[index], index));
        return output;
    }

    private static void copyRow(Tensor input, Tensor output, int sourceRow, int targetRow) {
        if (sourceRow < 0 || sourceRow >= input.rows) {
            throw new IllegalArgumentException("Row index out of range: " + sourceRow);
        }
        System.arraycopy(input.data, sourceRow * input.cols,
                output.data, targetRow * output.cols, input.cols);
    }

    private static void splitRow(Tensor input, Tensor output, int row, int heads, int headDim) {
        for (int head = 0; head < heads; head++) {
            int source = row * input.cols + head * headDim;
            int target = (head * input.rows + row) * headDim;
            System.arraycopy(input.data, source, output.data, target, headDim);
        }
    }

    static Tensor mergeHeads(Tensor input, int heads) {
        requireHeads(input.rows, heads);
        input.materialize();
        int rows = input.rows / heads;
        Tensor output = new Tensor(rows, heads * input.cols);
        DeepJExecutor.forRange(0, rows, row -> mergeRow(input, output, row, heads));
        return output;
    }

    private static void mergeRow(Tensor input, Tensor output, int row, int heads) {
        for (int head = 0; head < heads; head++) {
            int source = (head * output.rows + row) * input.cols;
            int target = row * output.cols + head * input.cols;
            System.arraycopy(input.data, source, output.data, target, input.cols);
        }
    }

    static Tensor batchedMatmul(Tensor left, Tensor right, int batches,
                                boolean transposeLeft, boolean transposeRight) {
        BatchShape shape = shape(left, right, batches, transposeLeft, transposeRight);
        left.materialize();
        right.materialize();
        Tensor output = new Tensor(batches * shape.rows(), shape.cols());
        DeepJExecutor.forRange(0, batches * shape.rows(), index ->
                multiplyRow(left, right, output, shape, index));
        return output;
    }

    private static void multiplyRow(Tensor left, Tensor right, Tensor output,
                                    BatchShape shape, int index) {
        int batch = index / shape.rows();
        int row = index % shape.rows();
        int outputBase = index * shape.cols();
        for (int inner = 0; inner < shape.inner(); inner++) {
            float value = leftValue(left, shape, batch, row, inner);
            for (int col = 0; col < shape.cols(); col++) {
                output.data[outputBase + col] += value * rightValue(right, shape, batch, inner, col);
            }
        }
    }

    private static float leftValue(Tensor tensor, BatchShape shape, int batch, int row, int inner) {
        int batchBase = batch * shape.leftRows() * tensor.cols;
        int offset = shape.transposeLeft() ? inner * tensor.cols + row : row * tensor.cols + inner;
        return tensor.data[batchBase + offset];
    }

    private static float rightValue(Tensor tensor, BatchShape shape, int batch, int inner, int col) {
        int batchBase = batch * shape.rightRows() * tensor.cols;
        int offset = shape.transposeRight() ? col * tensor.cols + inner : inner * tensor.cols + col;
        return tensor.data[batchBase + offset];
    }

    static Tensor causalMask(Tensor input, int sequenceLength) {
        requireCausalShape(input, sequenceLength);
        input.materialize();
        Tensor output = new Tensor(input.rows, input.cols);
        DeepJExecutor.forRange(0, input.rows, row -> maskRow(input, output, row, sequenceLength));
        return output;
    }

    private static void maskRow(Tensor input, Tensor output, int row, int sequenceLength) {
        int base = row * input.cols;
        int position = row % sequenceLength;
        for (int col = 0; col < input.cols; col++) {
            output.data[base + col] = col > position ? -1e9f : input.data[base + col];
        }
    }

    static Tensor rotary(Tensor input, Tensor cosine, Tensor sine,
                         int sequenceLength, boolean inverse) {
        requireRotaryShape(input, cosine, sine, sequenceLength);
        input.materialize();
        cosine.materialize();
        sine.materialize();
        Tensor output = new Tensor(input.rows, input.cols);
        DeepJExecutor.forRange(0, input.rows, row ->
                rotateRow(input, cosine, sine, output, row, sequenceLength, inverse));
        return output;
    }

    private static void rotateRow(Tensor input, Tensor cosine, Tensor sine, Tensor output,
                                  int row, int sequenceLength, boolean inverse) {
        int position = row % sequenceLength;
        for (int pair = 0; pair < input.cols / 2; pair++) {
            rotatePair(input, cosine, sine, output, row, position, pair, inverse);
        }
    }

    private static void rotatePair(Tensor input, Tensor cosine, Tensor sine, Tensor output,
                                   int row, int position, int pair, boolean inverse) {
        int inputBase = row * input.cols + pair * 2;
        int tableIndex = position * cosine.cols + pair;
        float direction = inverse ? -1.0f : 1.0f;
        float x = input.data[inputBase], y = input.data[inputBase + 1];
        float cos = cosine.data[tableIndex], sin = sine.data[tableIndex] * direction;
        output.data[inputBase] = x * cos - y * sin;
        output.data[inputBase + 1] = x * sin + y * cos;
    }

    private static BatchShape shape(Tensor left, Tensor right, int batches,
                                    boolean transposeLeft, boolean transposeRight) {
        requireHeads(left.rows, batches);
        requireHeads(right.rows, batches);
        int leftRows = left.rows / batches, rightRows = right.rows / batches;
        int rows = transposeLeft ? left.cols : leftRows;
        int innerLeft = transposeLeft ? leftRows : left.cols;
        int innerRight = transposeRight ? right.cols : rightRows;
        int cols = transposeRight ? rightRows : right.cols;
        if (innerLeft != innerRight) throw new IllegalArgumentException("Batched matmul shape mismatch");
        return new BatchShape(leftRows, rightRows, rows, cols, innerLeft,
                transposeLeft, transposeRight);
    }

    private static void requireHeads(int dimension, int heads) {
        if (heads <= 0 || dimension % heads != 0) {
            throw new IllegalArgumentException("Dimension must be divisible by a positive head count");
        }
    }

    private static void requireCausalShape(Tensor input, int sequenceLength) {
        if (sequenceLength <= 0 || input.cols != sequenceLength || input.rows % sequenceLength != 0) {
            throw new IllegalArgumentException("Scores must contain complete square attention heads");
        }
    }

    private static void requireRotaryShape(Tensor input, Tensor cosine, Tensor sine,
                                           int sequenceLength) {
        if (sequenceLength <= 0 || input.rows % sequenceLength != 0 || input.cols % 2 != 0) {
            throw new IllegalArgumentException("Invalid rotary input shape");
        }
        if (cosine.rows < sequenceLength || cosine.cols != input.cols / 2
                || sine.rows != cosine.rows || sine.cols != cosine.cols) {
            throw new IllegalArgumentException("Invalid rotary table shape");
        }
    }

    private record BatchShape(int leftRows, int rightRows, int rows, int cols, int inner,
                              boolean transposeLeft, boolean transposeRight) {}
}
