package io.github.kirstenali.deepj.tensor;

import java.lang.ref.WeakReference;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

abstract class GraphRecordingOps extends GraphCore {
    GraphRecordingOps(GpuRuntime runtime) {
        super(runtime);
    }
    public void recordBinary(int opCode, GpuBuffer a, GpuBuffer b, GpuBuffer out) {
        beginOp(5);
        emitInt(opCode);
        emitInt(a.id);
        emitInt(b.id);
        emitInt(out.id);
        emitInt(out.floatCount());
        endOp();
    }

    public void recordMatmul(GpuBuffer a, GpuBuffer b, GpuBuffer out, int m, int n, int k) {
        beginOp(7);
        emitInt(OP_MATMUL);
        emitInt(a.id);
        emitInt(b.id);
        emitInt(out.id);
        emitInt(m);
        emitInt(n);
        emitInt(k);
        endOp();
    }

    public void recordBatchedMatmul(GpuBuffer left, GpuBuffer right, GpuBuffer out,
                                    int batches, int leftRows, int leftCols,
                                    int rightRows, int rightCols,
                                    boolean transposeLeft, boolean transposeRight) {
        beginOp(11);
        emitInt(OP_BATCHED_MATMUL);
        emitInt(left.id);
        emitInt(right.id);
        emitInt(out.id);
        emitInt(batches);
        emitInt(leftRows);
        emitInt(leftCols);
        emitInt(rightRows);
        emitInt(rightCols);
        emitInt(transposeLeft ? 1 : 0);
        emitInt(transposeRight ? 1 : 0);
        endOp();
    }

    public void recordHeadPermutation(int opCode, GpuBuffer input, GpuBuffer output,
                                      int sequenceLength, int heads, int headDim, int modelWidth) {
        beginOp(7);
        emitInt(opCode);
        emitInt(input.id);
        emitInt(output.id);
        emitInt(sequenceLength);
        emitInt(heads);
        emitInt(headDim);
        emitInt(modelWidth);
        endOp();
    }

    public void recordCausalMask(GpuBuffer input, GpuBuffer output,
                                 int rows, int sequenceLength) {
        beginOp(5);
        emitInt(OP_CAUSAL_MASK);
        emitInt(input.id);
        emitInt(output.id);
        emitInt(rows);
        emitInt(sequenceLength);
        endOp();
    }

    public void recordCausalSoftmax(GpuBuffer input, GpuBuffer output,
                                    int rows, int sequenceLength, float scale) {
        beginOp(7);
        emitInt(OP_CAUSAL_SOFTMAX);
        emitInt(input.id);
        emitInt(output.id);
        emitInt(rows);
        emitInt(sequenceLength);
        emitInt(sequenceLength);
        emitFloatBits(scale);
        endOp();
    }

    public void recordGatherRows(GpuBuffer input, GpuBuffer indices, GpuBuffer output,
                                 int inputRows, int columns, int outputRows) {
        beginOp(7);
        emitInt(OP_GATHER_ROWS);
        emitInt(input.id);
        emitInt(indices.id);
        emitInt(output.id);
        emitInt(inputRows);
        emitInt(columns);
        emitInt(outputRows);
        endOp();
    }

    public void recordRotary(GpuBuffer input, GpuBuffer cosine, GpuBuffer sine,
                             GpuBuffer output, int rows, int sequenceLength,
                             int headDim, boolean inverse) {
        beginOp(9);
        emitInt(OP_ROTARY);
        emitInt(input.id);
        emitInt(cosine.id);
        emitInt(sine.id);
        emitInt(output.id);
        emitInt(rows);
        emitInt(sequenceLength);
        emitInt(headDim);
        emitInt(inverse ? 1 : 0);
        endOp();
    }

    public void recordUnary(int opCode, GpuBuffer in, GpuBuffer out) {
        beginOp(4);
        emitInt(opCode);
        emitInt(in.id);
        emitInt(out.id);
        emitInt(out.floatCount());
        endOp();
    }

    public void recordMultiplyScalar(GpuBuffer in, GpuBuffer out, float scalar) {
        beginOp(5);
        emitInt(OP_MULTIPLY_SCALAR);
        emitInt(in.id);
        emitInt(out.id);
        emitFloatBits(scalar);
        emitInt(out.floatCount());
        endOp();
    }

    public void recordScalarUnary(int opCode, GpuBuffer in, GpuBuffer out, float scalar) {
        beginOp(5);
        emitInt(opCode);
        emitInt(in.id);
        emitInt(out.id);
        emitFloatBits(scalar);
        emitInt(out.floatCount());
        endOp();
    }

    public void recordPow(GpuBuffer in, GpuBuffer out, float exponent) {
        beginOp(5);
        emitInt(OP_POW);
        emitInt(in.id);
        emitInt(out.id);
        emitFloatBits(exponent);
        emitInt(out.floatCount());
        endOp();
    }

    public void recordClamp(GpuBuffer in, GpuBuffer out, float min, float max) {
        beginOp(6);
        emitInt(OP_CLAMP);
        emitInt(in.id);
        emitInt(out.id);
        emitFloatBits(min);
        emitFloatBits(max);
        emitInt(out.floatCount());
        endOp();
    }

    public void recordScatterAddRows(GpuBuffer target, GpuBuffer indices, GpuBuffer grad,
                                     int targetRows, int targetCols, int nIndices) {
        beginOp(7);
        emitInt(OP_SCATTER_ADD_ROWS);
        emitInt(target.id);
        emitInt(indices.id);
        emitInt(grad.id);
        emitInt(targetRows);
        emitInt(targetCols);
        emitInt(nIndices);
        endOp();
    }

    public void recordScatterAddRowsAtomic(GpuBuffer target, GpuBuffer indices, GpuBuffer grad,
                                           int targetRows, int targetCols, int nIndices) {
        beginOp(7);
        emitInt(OP_SCATTER_ADD_ROWS_ATOMIC);
        emitInt(target.id);
        emitInt(indices.id);
        emitInt(grad.id);
        emitInt(targetRows);
        emitInt(targetCols);
        emitInt(nIndices);
        endOp();
    }

    public void recordSumAbs(GpuBuffer in, GpuBuffer out, int rows, int cols) {
        beginOp(5);
        emitInt(OP_SUM_ABS);
        emitInt(in.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordSumSquares(GpuBuffer in, GpuBuffer out, int rows, int cols) {
        beginOp(5);
        emitInt(OP_SUM_SQUARES);
        emitInt(in.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordSumSquaresScalar(GpuBuffer input, GpuBuffer total) {
        beginOp(4);
        emitInt(OP_SUM_SQUARES_SCALAR);
        emitInt(input.id);
        emitInt(total.id);
        emitInt(input.floatCount());
        endOp();
    }

    public void recordCrossEntropyLoss(GpuBuffer logits, GpuBuffer targets, GpuBuffer out, int rows, int cols) {
        beginOp(6);
        emitInt(OP_CROSS_ENTROPY_LOSS);
        emitInt(logits.id);
        emitInt(targets.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordCrossEntropyGradient(GpuBuffer logits, GpuBuffer targets, GpuBuffer out, int rows, int cols) {
        beginOp(6);
        emitInt(OP_CROSS_ENTROPY_GRADIENT);
        emitInt(logits.id);
        emitInt(targets.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }


}
