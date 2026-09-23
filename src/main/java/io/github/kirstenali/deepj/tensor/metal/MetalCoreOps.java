package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;

import java.util.List;

abstract class MetalCoreOps implements TensorBackend {
    final ComputeGraph graph;

    MetalCoreOps() {
        if (!isAvailable()) throw new IllegalStateException("Metal is not available on this system");
        this.graph = new ComputeGraph(new MetalGpuRuntime());
    }

    public static boolean isAvailable() {
        return MetalNative.AVAILABLE;
    }

    static void requireRowVector(Tensor rowVector, Tensor target) {
        if (rowVector.rows != 1 || rowVector.cols != target.cols) {
            throw new IllegalArgumentException(
                    "Expected row vector 1x" + target.cols + " but got " + rowVector.rows + "x" + rowVector.cols);
        }
    }

    static void requireColVector(Tensor colVector, Tensor target) {
        if (colVector.rows != target.rows || colVector.cols != 1) {
            throw new IllegalArgumentException(
                    "Expected col vector " + target.rows + "x1 but got " + colVector.rows + "x" + colVector.cols);
        }
    }

    GpuBuffer gpuIn(Tensor t) { return graph.ensureGpuBuffer(t); }

    Tensor gpuOut(GpuBuffer buf) {
        return graph.createOutputTensor(buf);
    }

    static void markNeedsUpload(Tensor t) {
        if (t.getGpuTag() instanceof GpuBuffer gb) {
            gb.needsUpload = true;
            gb.cpuStale = false;
        }
    }

    static Tensor immutableIntColumn(int[] values) {
        Tensor t = new Tensor(values.length, 1);
        for (int i = 0; i < values.length; i++) {
            t.data[i] = values[i];
        }
        return t;
    }

    @Override
    public void materializeTensor(Tensor t) {
        graph.materialize(t);
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        if (a.cols != b.rows) throw new IllegalArgumentException(
                "Shape mismatch for matmul: " + a.rows + "x" + a.cols + " vs " + b.rows + "x" + b.cols);
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, b.cols);
        graph.recordMatmul(ga, gb, gOut, a.rows, b.cols, a.cols);
        return gpuOut(gOut);
    }

    @Override
    public Tensor sliceRows(Tensor input, int[] rows) {
        for (int row : rows) {
            requireRow(input, row);
        }
        Tensor indices = immutableIntColumn(rows);
        GpuBuffer output = graph.newOutputBuffer(rows.length, input.cols);
        graph.recordGatherRows(gpuIn(input), gpuIn(indices), output,
                input.rows, input.cols, rows.length);
        return gpuOut(output);
    }

    static void requireRow(Tensor input, int row) {
        if (row < 0 || row >= input.rows) {
            throw new IllegalArgumentException("Row index out of range: " + row);
        }
    }

    @Override
    public Tensor splitHeads(Tensor input, int heads) {
        requireDivisible(input.cols, heads, "model width");
        int headDim = input.cols / heads;
        GpuBuffer output = graph.newOutputBuffer(heads * input.rows, headDim);
        graph.recordHeadPermutation(ComputeGraph.OP_SPLIT_HEADS, gpuIn(input), output,
                input.rows, heads, headDim, input.cols);
        return gpuOut(output);
    }

    @Override
    public Tensor mergeHeads(Tensor input, int heads) {
        requireDivisible(input.rows, heads, "head rows");
        int sequenceLength = input.rows / heads;
        int modelWidth = heads * input.cols;
        GpuBuffer output = graph.newOutputBuffer(sequenceLength, modelWidth);
        graph.recordHeadPermutation(ComputeGraph.OP_MERGE_HEADS, gpuIn(input), output,
                sequenceLength, heads, input.cols, modelWidth);
        return gpuOut(output);
    }

    @Override
    public Tensor batchedMatmul(Tensor left, Tensor right, int batches,
                                boolean transposeLeft, boolean transposeRight) {
        BatchedShape shape = batchedShape(left, right, batches, transposeLeft, transposeRight);
        GpuBuffer output = graph.newOutputBuffer(batches * shape.rows(), shape.cols());
        graph.recordBatchedMatmul(gpuIn(left), gpuIn(right), output, batches,
                shape.leftRows(), left.cols, shape.rightRows(), right.cols,
                transposeLeft, transposeRight);
        return gpuOut(output);
    }

    @Override
    public Tensor causalMask(Tensor input, int sequenceLength) {
        requireCausalShape(input, sequenceLength);
        GpuBuffer output = graph.newOutputBuffer(input.rows, input.cols);
        graph.recordCausalMask(gpuIn(input), output, input.rows, sequenceLength);
        return gpuOut(output);
    }

    @Override
    public Tensor causalSoftmax(Tensor input, int sequenceLength, float scale) {
        requireCausalShape(input, sequenceLength);
        GpuBuffer output = graph.newOutputBuffer(input.rows, input.cols);
        graph.recordCausalSoftmax(gpuIn(input), output, input.rows, sequenceLength, scale);
        return gpuOut(output);
    }

    @Override
    public Tensor rotary(Tensor input, Tensor cosine, Tensor sine,
                         int sequenceLength, boolean inverse) {
        requireRotaryShape(input, cosine, sine, sequenceLength);
        GpuBuffer output = graph.newOutputBuffer(input.rows, input.cols);
        graph.recordRotary(gpuIn(input), gpuIn(cosine), gpuIn(sine), output,
                input.rows, sequenceLength, input.cols, inverse);
        return gpuOut(output);
    }

    static BatchedShape batchedShape(Tensor left, Tensor right, int batches,
                                             boolean transposeLeft, boolean transposeRight) {
        requireDivisible(left.rows, batches, "left rows");
        requireDivisible(right.rows, batches, "right rows");
        int leftRows = left.rows / batches, rightRows = right.rows / batches;
        int rows = transposeLeft ? left.cols : leftRows;
        int leftInner = transposeLeft ? leftRows : left.cols;
        int rightInner = transposeRight ? right.cols : rightRows;
        int cols = transposeRight ? rightRows : right.cols;
        if (leftInner != rightInner) throw new IllegalArgumentException("Batched matmul shape mismatch");
        return new BatchedShape(leftRows, rightRows, rows, cols);
    }

    static void requireDivisible(int dimension, int divisor, String name) {
        if (divisor <= 0 || dimension % divisor != 0) {
            throw new IllegalArgumentException(name + " must be divisible by a positive batch count");
        }
    }

    static void requireCausalShape(Tensor input, int sequenceLength) {
        if (sequenceLength <= 0 || input.cols != sequenceLength
                || input.rows % sequenceLength != 0) {
            throw new IllegalArgumentException("Scores must contain complete square attention heads");
        }
    }

    static void requireRotaryShape(Tensor input, Tensor cosine, Tensor sine,
                                           int sequenceLength) {
        if (sequenceLength <= 0 || input.rows % sequenceLength != 0 || input.cols % 2 != 0) {
            throw new IllegalArgumentException("Invalid rotary input shape");
        }
        if (cosine.rows < sequenceLength || cosine.cols != input.cols / 2
                || sine.rows != cosine.rows || sine.cols != cosine.cols) {
            throw new IllegalArgumentException("Invalid rotary table shape");
        }
    }

    record BatchedShape(int leftRows, int rightRows, int rows, int cols) {}

    @Override
    public Tensor add(Tensor a, Tensor b) {
        Tensor.requireSameShape(a, b, "add");
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordBinary(ComputeGraph.OP_ADD, ga, gb, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor subtract(Tensor a, Tensor b) {
        Tensor.requireSameShape(a, b, "subtract");
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordBinary(ComputeGraph.OP_SUBTRACT, ga, gb, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor multiply(Tensor a, Tensor b) {
        Tensor.requireSameShape(a, b, "multiply");
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordBinary(ComputeGraph.OP_MULTIPLY, ga, gb, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor divide(Tensor a, Tensor b) {
        Tensor.requireSameShape(a, b, "divide");
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordBinary(ComputeGraph.OP_DIVIDE, ga, gb, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor addRowVector(Tensor a, Tensor v) {
        requireRowVector(v, a);
        GpuBuffer ga = gpuIn(a), gv = gpuIn(v);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordRowBroadcast(ComputeGraph.OP_ADD_ROW_VECTOR, ga, gv, gOut, a.rows, a.cols);
        return gpuOut(gOut);
    }

    @Override
    public Tensor addBroadcastCols(Tensor a, Tensor v) {
        requireColVector(v, a);
        GpuBuffer ga = gpuIn(a), gv = gpuIn(v);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordColBroadcast(ComputeGraph.OP_ADD_BROADCAST_COLS, ga, gv, gOut, a.rows, a.cols);
        return gpuOut(gOut);
    }

    @Override
    public Tensor divideBroadcastCols(Tensor a, Tensor v) {
        requireColVector(v, a);
        GpuBuffer ga = gpuIn(a), gv = gpuIn(v);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordColBroadcast(ComputeGraph.OP_DIVIDE_BROADCAST_COLS, ga, gv, gOut, a.rows, a.cols);
        return gpuOut(gOut);
    }

    @Override
    public Tensor subtractBroadcastCols(Tensor a, Tensor v) {
        requireColVector(v, a);
        GpuBuffer ga = gpuIn(a), gv = gpuIn(v);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordColBroadcast(ComputeGraph.OP_SUBTRACT_BROADCAST_COLS, ga, gv, gOut, a.rows, a.cols);
        return gpuOut(gOut);
    }

    @Override
    public Tensor multiplyBroadcastCols(Tensor a, Tensor v) {
        requireColVector(v, a);
        GpuBuffer ga = gpuIn(a), gv = gpuIn(v);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordColBroadcast(ComputeGraph.OP_MULTIPLY_BROADCAST_COLS, ga, gv, gOut, a.rows, a.cols);
        return gpuOut(gOut);
    }

    @Override
    public Tensor addBroadcastRows(Tensor a, Tensor v) {
        return addRowVector(a, v);
    }

    @Override
    public Tensor multiplyBroadcastRows(Tensor a, Tensor v) {
        requireRowVector(v, a);
        GpuBuffer ga = gpuIn(a), gv = gpuIn(v);
        GpuBuffer gOut = graph.newOutputBuffer(a.rows, a.cols);
        graph.recordRowBroadcast(ComputeGraph.OP_MULTIPLY_BROADCAST_ROWS, ga, gv, gOut, a.rows, a.cols);
        return gpuOut(gOut);
    }


}
