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
        if (rowVector.matrixRows() != 1 || rowVector.matrixCols() != target.matrixCols()) {
            throw new IllegalArgumentException(
                    "Expected row vector 1x" + target.matrixCols() + " but got "
                            + rowVector.matrixRows() + "x" + rowVector.matrixCols());
        }
    }

    static void requireColVector(Tensor colVector, Tensor target) {
        if (colVector.matrixRows() != target.matrixRows() || colVector.matrixCols() != 1) {
            throw new IllegalArgumentException(
                    "Expected col vector " + target.matrixRows() + "x1 but got "
                            + colVector.matrixRows() + "x" + colVector.matrixCols());
        }
    }

    GpuBuffer gpuIn(Tensor t) { return graph.ensureGpuBuffer(t); }

    Tensor gpuOut(GpuBuffer buf) {
        return graph.createOutputTensor(buf);
    }

    @Override
    public Tensor zeros(int rows, int columns) {
        if (rows == 0 || columns == 0) return graph.zeroToDevice(DType.FLOAT32, rows, columns);
        return gpuOut(graph.newZeroBuffer(rows, columns));
    }

    @Override
    public Tensor zeros(DType dtype, int... shape) {
        return graph.zeroToDevice(dtype, shape);
    }

    @Override
    public Tensor copy(Tensor source) {
        return graph.copyToDevice(source);
    }

    @Override
    public Tensor cast(Tensor input, DType dtype) {
        return graph.castToDevice(input, dtype);
    }

    @Override
    public void copyInto(Tensor target, Tensor source) {
        graph.assignToDevice(target, source);
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
            t.rawData()[i] = values[i];
        }
        return t;
    }

    @Override
    public void materializeTensor(Tensor t) {
        graph.materialize(t);
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        if (a.matrixCols() != b.matrixRows()) throw new IllegalArgumentException(
                "Shape mismatch for matmul: " + a.matrixRows() + "x" + a.matrixCols()
                        + " vs " + b.matrixRows() + "x" + b.matrixCols());
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), b.matrixCols());
        graph.recordMatmul(ga, gb, gOut, a.matrixRows(), b.matrixCols(), a.matrixCols());
        return gpuOut(gOut);
    }

    @Override
    public Tensor sliceRows(Tensor input, int[] rows) {
        for (int row : rows) {
            requireRow(input, row);
        }
        Tensor indices = immutableIntColumn(rows);
        GpuBuffer output = graph.newOutputBuffer(rows.length, input.matrixCols());
        graph.recordGatherRows(gpuIn(input), gpuIn(indices), output,
                input.matrixRows(), input.matrixCols(), rows.length);
        return gpuOut(output);
    }

    static void requireRow(Tensor input, int row) {
        if (row < 0 || row >= input.matrixRows()) {
            throw new IllegalArgumentException("Row index out of range: " + row);
        }
    }

    @Override
    public Tensor splitHeads(Tensor input, int heads) {
        requireDivisible(input.matrixCols(), heads, "model width");
        int headDim = input.matrixCols() / heads;
        GpuBuffer output = graph.newOutputBuffer(heads * input.matrixRows(), headDim);
        graph.recordHeadPermutation(ComputeGraph.OP_SPLIT_HEADS, gpuIn(input), output,
                input.matrixRows(), heads, headDim, input.matrixCols());
        return gpuOut(output);
    }

    @Override
    public Tensor mergeHeads(Tensor input, int heads) {
        requireDivisible(input.matrixRows(), heads, "head rows");
        int sequenceLength = input.matrixRows() / heads;
        int modelWidth = heads * input.matrixCols();
        GpuBuffer output = graph.newOutputBuffer(sequenceLength, modelWidth);
        graph.recordHeadPermutation(ComputeGraph.OP_MERGE_HEADS, gpuIn(input), output,
                sequenceLength, heads, input.matrixCols(), modelWidth);
        return gpuOut(output);
    }

    @Override
    public Tensor batchedMatmul(Tensor left, Tensor right, int batches,
                                boolean transposeLeft, boolean transposeRight) {
        BatchedShape shape = batchedShape(left, right, batches, transposeLeft, transposeRight);
        GpuBuffer output = graph.newOutputBuffer(batches * shape.rows(), shape.cols());
        graph.recordBatchedMatmul(gpuIn(left), gpuIn(right), output, batches,
                shape.leftRows(), left.matrixCols(), shape.rightRows(), right.matrixCols(),
                transposeLeft, transposeRight);
        return gpuOut(output);
    }

    @Override
    public Tensor causalMask(Tensor input, int sequenceLength) {
        MatrixShapeChecks.requireCausal(input, sequenceLength);
        GpuBuffer output = graph.newOutputBuffer(input.matrixRows(), input.matrixCols());
        graph.recordCausalMask(gpuIn(input), output, input.matrixRows(), sequenceLength);
        return gpuOut(output);
    }

    @Override
    public Tensor causalSoftmax(Tensor input, int sequenceLength, float scale) {
        MatrixShapeChecks.requireCausal(input, sequenceLength);
        GpuBuffer output = graph.newOutputBuffer(input.matrixRows(), input.matrixCols());
        graph.recordCausalSoftmax(gpuIn(input), output, input.matrixRows(), sequenceLength, scale);
        return gpuOut(output);
    }

    @Override
    public Tensor rotary(Tensor input, Tensor cosine, Tensor sine,
                         int sequenceLength, boolean inverse) {
        MatrixShapeChecks.requireRotary(input, cosine, sine, sequenceLength);
        GpuBuffer output = graph.newOutputBuffer(input.matrixRows(), input.matrixCols());
        graph.recordRotary(gpuIn(input), gpuIn(cosine), gpuIn(sine), output,
                input.matrixRows(), sequenceLength, input.matrixCols(), inverse);
        return gpuOut(output);
    }

    static BatchedShape batchedShape(Tensor left, Tensor right, int batches,
                                             boolean transposeLeft, boolean transposeRight) {
        requireDivisible(left.matrixRows(), batches, "left rows");
        requireDivisible(right.matrixRows(), batches, "right rows");
        int leftRows = left.matrixRows() / batches, rightRows = right.matrixRows() / batches;
        int rows = transposeLeft ? left.matrixCols() : leftRows;
        int leftInner = transposeLeft ? leftRows : left.matrixCols();
        int rightInner = transposeRight ? right.matrixCols() : rightRows;
        int cols = transposeRight ? rightRows : right.matrixCols();
        if (leftInner != rightInner) throw new IllegalArgumentException("Batched matmul shape mismatch");
        return new BatchedShape(leftRows, rightRows, rows, cols);
    }

    static void requireDivisible(int dimension, int divisor, String name) {
        if (divisor <= 0 || dimension % divisor != 0) {
            throw new IllegalArgumentException(name + " must be divisible by a positive batch count");
        }
    }

    record BatchedShape(int leftRows, int rightRows, int rows, int cols) {}

    @Override
    public Tensor add(Tensor a, Tensor b) {
        if (!MetalDenseOps.sameMatrices(a, b)) return graph.layoutBinary(a, b, ComputeGraph.OP_ADD);
        Tensor.requireSameShape(a, b, "add");
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordBinary(ComputeGraph.OP_ADD, ga, gb, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor subtract(Tensor a, Tensor b) {
        if (!MetalDenseOps.sameMatrices(a, b)) return graph.layoutBinary(a, b, ComputeGraph.OP_SUBTRACT);
        Tensor.requireSameShape(a, b, "subtract");
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordBinary(ComputeGraph.OP_SUBTRACT, ga, gb, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor multiply(Tensor a, Tensor b) {
        if (!MetalDenseOps.sameMatrices(a, b)) return graph.layoutBinary(a, b, ComputeGraph.OP_MULTIPLY);
        Tensor.requireSameShape(a, b, "multiply");
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordBinary(ComputeGraph.OP_MULTIPLY, ga, gb, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor divide(Tensor a, Tensor b) {
        if (!MetalDenseOps.sameMatrices(a, b)) return graph.layoutBinary(a, b, ComputeGraph.OP_DIVIDE);
        Tensor.requireSameShape(a, b, "divide");
        GpuBuffer ga = gpuIn(a), gb = gpuIn(b);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordBinary(ComputeGraph.OP_DIVIDE, ga, gb, gOut);
        return gpuOut(gOut);
    }

    @Override
    public Tensor addRowVector(Tensor a, Tensor v) {
        requireRowVector(v, a);
        if (!MetalDenseOps.matrix(a) || !MetalDenseOps.matrix(v)) return graph.layoutBinary(a, v, ComputeGraph.OP_ADD);
        GpuBuffer ga = gpuIn(a), gv = gpuIn(v);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordRowBroadcast(ComputeGraph.OP_ADD_ROW_VECTOR, ga, gv, gOut, a.matrixRows(), a.matrixCols());
        return gpuOut(gOut);
    }

    @Override
    public Tensor addBroadcastCols(Tensor a, Tensor v) {
        requireColVector(v, a);
        if (!MetalDenseOps.matrix(a) || !MetalDenseOps.matrix(v)) return graph.layoutBinary(a, v, ComputeGraph.OP_ADD);
        GpuBuffer ga = gpuIn(a), gv = gpuIn(v);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordColBroadcast(ComputeGraph.OP_ADD_BROADCAST_COLS, ga, gv, gOut, a.matrixRows(), a.matrixCols());
        return gpuOut(gOut);
    }

    @Override
    public Tensor divideBroadcastCols(Tensor a, Tensor v) {
        requireColVector(v, a);
        if (!MetalDenseOps.matrix(a) || !MetalDenseOps.matrix(v)) {
            return graph.layoutBinary(a, v, ComputeGraph.OP_DIVIDE);
        }
        GpuBuffer ga = gpuIn(a), gv = gpuIn(v);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordColBroadcast(ComputeGraph.OP_DIVIDE_BROADCAST_COLS, ga, gv, gOut, a.matrixRows(), a.matrixCols());
        return gpuOut(gOut);
    }

    @Override
    public Tensor subtractBroadcastCols(Tensor a, Tensor v) {
        requireColVector(v, a);
        if (!MetalDenseOps.matrix(a) || !MetalDenseOps.matrix(v)) {
            return graph.layoutBinary(a, v, ComputeGraph.OP_SUBTRACT);
        }
        GpuBuffer ga = gpuIn(a), gv = gpuIn(v);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordColBroadcast(ComputeGraph.OP_SUBTRACT_BROADCAST_COLS, ga, gv, gOut, a.matrixRows(), a.matrixCols());
        return gpuOut(gOut);
    }

    @Override
    public Tensor multiplyBroadcastCols(Tensor a, Tensor v) {
        requireColVector(v, a);
        if (!MetalDenseOps.matrix(a) || !MetalDenseOps.matrix(v)) {
            return graph.layoutBinary(a, v, ComputeGraph.OP_MULTIPLY);
        }
        GpuBuffer ga = gpuIn(a), gv = gpuIn(v);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordColBroadcast(ComputeGraph.OP_MULTIPLY_BROADCAST_COLS, ga, gv, gOut, a.matrixRows(), a.matrixCols());
        return gpuOut(gOut);
    }

    @Override
    public Tensor addBroadcastRows(Tensor a, Tensor v) {
        return addRowVector(a, v);
    }

    @Override
    public Tensor multiplyBroadcastRows(Tensor a, Tensor v) {
        requireRowVector(v, a);
        if (!MetalDenseOps.matrix(a) || !MetalDenseOps.matrix(v)) {
            return graph.layoutBinary(a, v, ComputeGraph.OP_MULTIPLY);
        }
        GpuBuffer ga = gpuIn(a), gv = gpuIn(v);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), a.matrixCols());
        graph.recordRowBroadcast(ComputeGraph.OP_MULTIPLY_BROADCAST_ROWS, ga, gv, gOut, a.matrixRows(), a.matrixCols());
        return gpuOut(gOut);
    }


}
