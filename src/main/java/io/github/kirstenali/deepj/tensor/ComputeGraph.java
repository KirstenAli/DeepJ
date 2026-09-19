package io.github.kirstenali.deepj.tensor;

import java.lang.ref.WeakReference;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public final class ComputeGraph {

    // Keep these values synchronized with deepj_metal_jni.mm.
    public static final int OP_ADD             = 1;
    public static final int OP_SUBTRACT        = 2;
    public static final int OP_MULTIPLY        = 3;
    public static final int OP_DIVIDE          = 4;
    public static final int OP_MATMUL          = 5;
    public static final int OP_MULTIPLY_SCALAR = 6;
    public static final int OP_SQRT            = 7;
    public static final int OP_NEG             = 8;
    public static final int OP_EXP             = 9;
    public static final int OP_LOG             = 10;
    public static final int OP_TANH            = 11;
    public static final int OP_SIGMOID         = 12;
    public static final int OP_RELU            = 13;
    public static final int OP_RELU_BACKWARD   = 14;
    public static final int OP_GELU            = 15;
    public static final int OP_GELU_BACKWARD   = 16;
    public static final int OP_SOFTMAX_ROWS    = 17;
    public static final int OP_SOFTMAX_BACKWARD= 18;
    public static final int OP_LAYERNORM_BACKWARD = 19;
    public static final int OP_ADAMW_UPDATE    = 20;
    public static final int OP_ADD_SCALAR      = 21;
    public static final int OP_DIVIDE_SCALAR   = 22;
    public static final int OP_TRANSPOSE       = 23;
    public static final int OP_ADD_ROW_VECTOR  = 24;
    public static final int OP_ADD_BROADCAST_COLS = 25;
    public static final int OP_SUBTRACT_BROADCAST_COLS = 26;
    public static final int OP_DIVIDE_BROADCAST_COLS = 27;
    public static final int OP_MULTIPLY_BROADCAST_ROWS = 28;
    public static final int OP_SUM_ROWS        = 29;
    public static final int OP_MEAN_ALONG_ROWS = 30;
    public static final int OP_VARIANCE_ALONG_ROWS = 31;
    public static final int OP_MULTIPLY_BROADCAST_COLS = 32;
    public static final int OP_SUM_ALONG_ROWS = 33;
    public static final int OP_MAX_ALONG_ROWS = 34;
    public static final int OP_CLAMP = 35;
    public static final int OP_POW = 36;
    public static final int OP_SCATTER_ADD_ROWS = 37;
    public static final int OP_SUM_ABS = 38;
    public static final int OP_CROSS_ENTROPY_LOSS = 39;
    public static final int OP_CROSS_ENTROPY_GRADIENT = 40;
    public static final int OP_SUM_SCALAR = 41;
    public static final int OP_SCATTER_ADD_ROWS_ATOMIC = 42;
    public static final int OP_SUM_SQUARES = 43;
    public static final int OP_BATCHED_MATMUL = 44;
    public static final int OP_SPLIT_HEADS = 45;
    public static final int OP_MERGE_HEADS = 46;
    public static final int OP_CAUSAL_MASK = 47;
    public static final int OP_ROTARY = 48;
    public static final int OP_GATHER_ROWS = 49;
    public static final int OP_CAUSAL_SOFTMAX = 50;

    private record OpMeta(int stride, int[] bufferArgOffsets) {}

    private static final OpMeta[] OP_METADATA = buildOpMetadata();

    private final GpuRuntime runtime;

    private static final AtomicInteger NEXT_BUFFER_ID = new AtomicInteger();

    private int[] cmdStream = new int[4096];
    private int cmdPos = 0;
    private int opCount = 0;

    private final List<int[]> pendingAllocs = new ArrayList<>();
    private final List<int[]> pendingUploadIds = new ArrayList<>();
    private final List<float[]> pendingUploadData = new ArrayList<>();
    private final Map<Integer, WeakReference<Tensor>> bufIdToTensor = new HashMap<>();
    private final Set<Integer> allocatedBufferIds = new HashSet<>();

    public ComputeGraph(GpuRuntime runtime) {
        this.runtime = Objects.requireNonNull(runtime, "runtime");
    }

    private static OpMeta[] buildOpMetadata() {
        OpMeta[] meta = new OpMeta[OP_CAUSAL_SOFTMAX + 1];
        registerUnaryMeta(meta);
        registerBinaryMeta(meta);
        registerReductionMeta(meta);
        registerLossMeta(meta);
        registerAttentionMeta(meta);
        registerBroadcastMeta(meta);
        registerComplexMeta(meta);
        return meta;
    }

    private static void registerUnaryMeta(OpMeta[] meta) {
        registerMeta(meta, OP_SQRT, 4, 1, 2);
        registerMeta(meta, OP_NEG, 4, 1, 2);
        registerMeta(meta, OP_EXP, 4, 1, 2);
        registerMeta(meta, OP_LOG, 4, 1, 2);
        registerMeta(meta, OP_TANH, 4, 1, 2);
        registerMeta(meta, OP_SIGMOID, 4, 1, 2);
        registerMeta(meta, OP_RELU, 4, 1, 2);
        registerMeta(meta, OP_GELU, 4, 1, 2);
    }

    private static void registerBinaryMeta(OpMeta[] meta) {
        registerMeta(meta, OP_ADD, 5, 1, 2, 3);
        registerMeta(meta, OP_SUBTRACT, 5, 1, 2, 3);
        registerMeta(meta, OP_MULTIPLY, 5, 1, 2, 3);
        registerMeta(meta, OP_DIVIDE, 5, 1, 2, 3);
        registerMeta(meta, OP_RELU_BACKWARD, 5, 1, 2, 3);
        registerMeta(meta, OP_GELU_BACKWARD, 5, 1, 2, 3);
    }

    private static void registerReductionMeta(OpMeta[] meta) {
        registerMeta(meta, OP_MULTIPLY_SCALAR, 5, 1, 2);
        registerMeta(meta, OP_ADD_SCALAR, 5, 1, 2);
        registerMeta(meta, OP_DIVIDE_SCALAR, 5, 1, 2);
        registerMeta(meta, OP_TRANSPOSE, 5, 1, 2);
        registerMeta(meta, OP_SUM_ROWS, 5, 1, 2);
        registerMeta(meta, OP_MEAN_ALONG_ROWS, 5, 1, 2);
        registerMeta(meta, OP_VARIANCE_ALONG_ROWS, 5, 1, 2);
        registerMeta(meta, OP_SUM_ALONG_ROWS, 5, 1, 2);
        registerMeta(meta, OP_MAX_ALONG_ROWS, 5, 1, 2);
        registerMeta(meta, OP_SOFTMAX_ROWS, 5, 1, 2);
        registerMeta(meta, OP_CLAMP, 6, 1, 2);
        registerMeta(meta, OP_POW, 5, 1, 2);
        registerMeta(meta, OP_SCATTER_ADD_ROWS, 7, 1, 2, 3);
        registerMeta(meta, OP_SUM_ABS, 5, 1, 2);
        registerMeta(meta, OP_SUM_SQUARES, 5, 1, 2);
        registerMeta(meta, OP_SUM_SCALAR, 5, 1, 2);
    }

    private static void registerLossMeta(OpMeta[] meta) {
        registerMeta(meta, OP_CROSS_ENTROPY_LOSS, 6, 1, 2, 3);
        registerMeta(meta, OP_CROSS_ENTROPY_GRADIENT, 6, 1, 2, 3);
        registerMeta(meta, OP_SCATTER_ADD_ROWS_ATOMIC, 7, 1, 2, 3);
    }

    private static void registerAttentionMeta(OpMeta[] meta) {
        registerMeta(meta, OP_BATCHED_MATMUL, 11, 1, 2, 3);
        registerMeta(meta, OP_SPLIT_HEADS, 7, 1, 2);
        registerMeta(meta, OP_MERGE_HEADS, 7, 1, 2);
        registerMeta(meta, OP_CAUSAL_MASK, 5, 1, 2);
        registerMeta(meta, OP_ROTARY, 9, 1, 2, 3, 4);
        registerMeta(meta, OP_GATHER_ROWS, 7, 1, 2, 3);
        registerMeta(meta, OP_CAUSAL_SOFTMAX, 7, 1, 2);
    }

    private static void registerBroadcastMeta(OpMeta[] meta) {
        registerMeta(meta, OP_SOFTMAX_BACKWARD, 6, 1, 2, 3);
        registerMeta(meta, OP_ADD_ROW_VECTOR, 6, 1, 2, 3);
        registerMeta(meta, OP_ADD_BROADCAST_COLS, 6, 1, 2, 3);
        registerMeta(meta, OP_SUBTRACT_BROADCAST_COLS, 6, 1, 2, 3);
        registerMeta(meta, OP_DIVIDE_BROADCAST_COLS, 6, 1, 2, 3);
        registerMeta(meta, OP_MULTIPLY_BROADCAST_ROWS, 6, 1, 2, 3);
        registerMeta(meta, OP_MULTIPLY_BROADCAST_COLS, 6, 1, 2, 3);
    }

    private static void registerComplexMeta(OpMeta[] meta) {
        registerMeta(meta, OP_MATMUL, 7, 1, 2, 3);
        registerMeta(meta, OP_LAYERNORM_BACKWARD, 7, 1, 2, 3, 4);
        registerMeta(meta, OP_ADAMW_UPDATE, 13, 1, 2, 3, 4);
    }

    private static void registerMeta(OpMeta[] meta, int op, int stride, int... bufferArgOffsets) {
        meta[op] = new OpMeta(stride, bufferArgOffsets);
    }

    private void scheduleAlloc(int id, int floatCount) {
        pendingAllocs.add(new int[]{id, floatCount});
        allocatedBufferIds.add(id);
    }

    private void scheduleUpload(int id, float[] data) {
        pendingUploadIds.add(new int[]{id});
        pendingUploadData.add(data);
    }

    public GpuBuffer ensureGpuBuffer(Tensor t) {
        if (t.getGpuTag() instanceof GpuBuffer existing) {
            return reuseExistingInputBuffer(t, existing);
        }

        return createAndUploadInputBuffer(t);
    }

    private GpuBuffer reuseExistingInputBuffer(Tensor t, GpuBuffer existing) {
        trackTensorBinding(existing.id, t);
        if (existing.needsUpload) {
            scheduleUpload(existing.id, TensorAdapters.packF32(t));
            existing.needsUpload = false;
            existing.cpuStale = false;
        }
        return existing;
    }

    private GpuBuffer createAndUploadInputBuffer(Tensor t) {
        int id = nextBufferId();
        GpuBuffer buf = new GpuBuffer(id, t.rows, t.cols, true);
        buf.allocatedOnGpu = false;

        scheduleAlloc(id, buf.floatCount());
        scheduleUpload(id, TensorAdapters.packF32(t));
        buf.needsUpload = false;
        buf.cpuStale = false;

        t.setGpuTag(buf);
        trackTensorBinding(id, t);
        return buf;
    }

    private void trackTensorBinding(int id, Tensor t) {
        bufIdToTensor.put(id, new WeakReference<>(t));
    }

    public GpuBuffer newOutputBuffer(int rows, int cols) {
        int id = nextBufferId();
        GpuBuffer buf = new GpuBuffer(id, rows, cols, false);
        buf.cpuStale = true;
        buf.allocatedOnGpu = false;
        scheduleAlloc(id, buf.floatCount());
        return buf;
    }

    public Tensor createOutputTensor(GpuBuffer buf) {
        Tensor t = new Tensor(buf.rows, buf.cols);
        bindTensorToBuffer(t, buf);
        return t;
    }

    public void bindTensorToBuffer(Tensor t, GpuBuffer buf) {
        t.setGpuTag(buf);
        trackTensorBinding(buf.id, t);
    }

    private void ensureCapacity(int needed) {
        if (cmdPos + needed > cmdStream.length) {
            cmdStream = Arrays.copyOf(cmdStream, Math.max(cmdStream.length * 2, cmdPos + needed));
        }
    }

    private void beginOp(int encodedInts) {
        ensureCapacity(encodedInts);
    }

    private void emitInt(int value) {
        cmdStream[cmdPos++] = value;
    }

    private void emitFloatBits(float value) {
        emitInt(Float.floatToRawIntBits(value));
    }

    private void endOp() {
        opCount++;
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
        emitInt(left.id); emitInt(right.id); emitInt(out.id); emitInt(batches);
        emitInt(leftRows); emitInt(leftCols); emitInt(rightRows); emitInt(rightCols);
        emitInt(transposeLeft ? 1 : 0); emitInt(transposeRight ? 1 : 0);
        endOp();
    }

    public void recordHeadPermutation(int opCode, GpuBuffer input, GpuBuffer output,
                                      int sequenceLength, int heads, int headDim, int modelWidth) {
        beginOp(7);
        emitInt(opCode); emitInt(input.id); emitInt(output.id);
        emitInt(sequenceLength); emitInt(heads); emitInt(headDim); emitInt(modelWidth);
        endOp();
    }

    public void recordCausalMask(GpuBuffer input, GpuBuffer output,
                                 int rows, int sequenceLength) {
        beginOp(5);
        emitInt(OP_CAUSAL_MASK); emitInt(input.id); emitInt(output.id);
        emitInt(rows); emitInt(sequenceLength);
        endOp();
    }

    public void recordCausalSoftmax(GpuBuffer input, GpuBuffer output,
                                    int rows, int sequenceLength, float scale) {
        beginOp(7);
        emitInt(OP_CAUSAL_SOFTMAX); emitInt(input.id); emitInt(output.id);
        emitInt(rows); emitInt(sequenceLength); emitInt(sequenceLength);
        emitFloatBits(scale);
        endOp();
    }

    public void recordGatherRows(GpuBuffer input, GpuBuffer indices, GpuBuffer output,
                                 int inputRows, int columns, int outputRows) {
        beginOp(7);
        emitInt(OP_GATHER_ROWS); emitInt(input.id); emitInt(indices.id); emitInt(output.id);
        emitInt(inputRows); emitInt(columns); emitInt(outputRows);
        endOp();
    }

    public void recordRotary(GpuBuffer input, GpuBuffer cosine, GpuBuffer sine,
                             GpuBuffer output, int rows, int sequenceLength,
                             int headDim, boolean inverse) {
        beginOp(9);
        emitInt(OP_ROTARY); emitInt(input.id); emitInt(cosine.id); emitInt(sine.id);
        emitInt(output.id); emitInt(rows); emitInt(sequenceLength); emitInt(headDim);
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

    public void recordSumScalar(GpuBuffer in, GpuBuffer out, int rows, int cols) {
        beginOp(5);
        emitInt(OP_SUM_SCALAR);
        emitInt(in.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordTranspose(GpuBuffer in, GpuBuffer out, int rows, int cols) {
        beginOp(5);
        emitInt(OP_TRANSPOSE);
        emitInt(in.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordRowBroadcast(int opCode, GpuBuffer a, GpuBuffer rowVec, GpuBuffer out, int rows, int cols) {
        beginOp(6);
        emitInt(opCode);
        emitInt(a.id);
        emitInt(rowVec.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordColBroadcast(int opCode, GpuBuffer a, GpuBuffer colVec, GpuBuffer out, int rows, int cols) {
        beginOp(6);
        emitInt(opCode);
        emitInt(a.id);
        emitInt(colVec.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordReduction(int opCode, GpuBuffer in, GpuBuffer out, int rows, int cols) {
        beginOp(5);
        emitInt(opCode);
        emitInt(in.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordSoftmaxRows(GpuBuffer in, GpuBuffer out, int rows, int cols) {
        beginOp(5);
        emitInt(OP_SOFTMAX_ROWS);
        emitInt(in.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordSoftmaxBackward(GpuBuffer gradOutput, GpuBuffer softmaxOut, GpuBuffer out, int rows, int cols) {
        beginOp(6);
        emitInt(OP_SOFTMAX_BACKWARD);
        emitInt(gradOutput.id);
        emitInt(softmaxOut.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordLayerNormBackward(GpuBuffer dXHat, GpuBuffer xHat, GpuBuffer std, GpuBuffer out, int rows, int cols) {
        beginOp(7);
        emitInt(OP_LAYERNORM_BACKWARD);
        emitInt(dXHat.id);
        emitInt(xHat.id);
        emitInt(std.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordAdamWUpdate(GpuBuffer w, GpuBuffer g, GpuBuffer mt, GpuBuffer vt,
                                  float lr, float beta1, float beta2, float eps,
                                  float weightDecay, float bc1, float bc2, int n) {
        beginOp(13);
        emitInt(OP_ADAMW_UPDATE);
        emitInt(w.id);
        emitInt(g.id);
        emitInt(mt.id);
        emitInt(vt.id);
        emitFloatBits(lr);
        emitFloatBits(beta1);
        emitFloatBits(beta2);
        emitFloatBits(eps);
        emitFloatBits(weightDecay);
        emitFloatBits(bc1);
        emitFloatBits(bc2);
        emitInt(n);
        endOp();
    }

    public boolean isEmpty() { return opCount == 0; }

    public void flush() {
        if (opCount == 0 && pendingAllocs.isEmpty()) {
            releaseOrphanedBuffers();
            return;
        }

        allocatePendingBuffers();
        uploadPendingData();
        executePendingOps();
        releaseOrphanedBuffers();
    }

    private void allocatePendingBuffers() {
        if (pendingAllocs.isEmpty()) return;

        int count = pendingAllocs.size();
        int[] ids   = new int[count];
        int[] sizes = new int[count];
        for (int i = 0; i < count; i++) {
            ids[i]   = pendingAllocs.get(i)[0];
            sizes[i] = pendingAllocs.get(i)[1];
        }
        runtime.allocBuffers(ids, sizes, count);
        markAllocatedBuffers();
        pendingAllocs.clear();
    }

    private void markAllocatedBuffers() {
        for (var ref : bufIdToTensor.values()) {
            Tensor t = ref.get();
            if (t != null && t.getGpuTag() instanceof GpuBuffer gb) {
                gb.allocatedOnGpu = true;
            }
        }
    }

    private void releaseOrphanedBuffers() {
        if (bufIdToTensor.isEmpty()) return;

        List<Integer> orphanIds = collectOrphanIds();
        if (orphanIds.isEmpty()) return;

        for (int id : orphanIds) {
            removePendingForId(id);
            bufIdToTensor.remove(id);
            allocatedBufferIds.remove(id);
        }

        int[] ids = orphanIds.stream().mapToInt(Integer::intValue).toArray();
        runtime.releaseBuffers(ids, ids.length);
    }

    private List<Integer> collectOrphanIds() {
        List<Integer> orphanIds = new ArrayList<>();
        for (var entry : bufIdToTensor.entrySet()) {
            int id = entry.getKey();
            Tensor t = entry.getValue().get();
            if (isOrphanedBuffer(id, t) && !isBufferReferencedByPendingOps(id)) {
                orphanIds.add(id);
            }
        }
        return orphanIds;
    }

    private boolean isOrphanedBuffer(int id, Tensor t) {
        if (t == null) return true;
        return !(t.getGpuTag() instanceof GpuBuffer gb) || gb.id != id;
    }

    private static int getOpStride(int op) {
        if (op < 0 || op >= OP_METADATA.length) return -1;
        OpMeta meta = OP_METADATA[op];
        return meta == null ? -1 : meta.stride();
    }

    private boolean opReferencesBuffer(int pos, int op, int bufferId) {
        if (op < 0 || op >= OP_METADATA.length) return false;
        OpMeta meta = OP_METADATA[op];
        if (meta == null) return false;

        for (int offset : meta.bufferArgOffsets()) {
            if (cmdStream[pos + offset] == bufferId) return true;
        }
        return false;
    }

    private boolean isBufferReferencedByPendingOps(int bufferId) {
        int pos = 0;
        while (pos < cmdPos) {
            int op = cmdStream[pos];
            int stride = getOpStride(op);
            if (stride < 0) return true;
            if (opReferencesBuffer(pos, op, bufferId)) return true;
            pos += stride;
        }
        return false;
    }

    private void removePendingForId(int id) {
        for (int i = pendingAllocs.size() - 1; i >= 0; i--) {
            if (pendingAllocs.get(i)[0] == id) {
                pendingAllocs.remove(i);
            }
        }
        for (int i = pendingUploadIds.size() - 1; i >= 0; i--) {
            if (pendingUploadIds.get(i)[0] == id) {
                pendingUploadIds.remove(i);
                pendingUploadData.remove(i);
            }
        }
    }

    private void uploadPendingData() {
        if (pendingUploadIds.isEmpty()) return;

        for (int i = 0; i < pendingUploadIds.size(); i++) {
            runtime.uploadBuffer(pendingUploadIds.get(i)[0], pendingUploadData.get(i));
        }
        pendingUploadIds.clear();
        pendingUploadData.clear();
    }

    private void executePendingOps() {
        if (opCount > 0) {
            runtime.flushOps(cmdStream, cmdPos);
        }
        cmdPos  = 0;
        opCount = 0;
    }

    public void materialize(Tensor t) {
        if (!(t.getGpuTag() instanceof GpuBuffer buf)) return;
        if (!buf.cpuStale) return;

        flush();

        runtime.downloadBuffer(buf.id, t.data);
        buf.cpuStale = false;
    }

    public void releaseAll() {
        materializeTrackedTensors();
        clearTensorGpuTags();
        releaseNativeBuffers();
        resetGraphState();
    }

    public void releaseTemporary() {
        flush();
        List<Integer> ids = temporaryBufferIds();
        clearTensorGpuTags(ids);
        releaseNativeBuffers(ids);
        allocatedBufferIds.removeAll(ids);
    }

    private List<Integer> temporaryBufferIds() {
        List<Integer> ids = new ArrayList<>();
        for (int id : allocatedBufferIds) {
            WeakReference<Tensor> reference = bufIdToTensor.get(id);
            Tensor tensor = reference == null ? null : reference.get();
            if (tensor == null || !tensor.retainsDeviceBuffer()) ids.add(id);
        }
        return ids;
    }

    private void clearTensorGpuTags(List<Integer> ids) {
        for (int id : ids) {
            WeakReference<Tensor> reference = bufIdToTensor.remove(id);
            Tensor tensor = reference == null ? null : reference.get();
            if (ownsBuffer(tensor, id)) tensor.setGpuTag(null);
        }
    }

    private static boolean ownsBuffer(Tensor tensor, int id) {
        return tensor != null && tensor.getGpuTag() instanceof GpuBuffer buffer
                && buffer.id == id;
    }

    private void materializeTrackedTensors() {
        flush();
        for (WeakReference<Tensor> ref : bufIdToTensor.values()) {
            Tensor t = ref.get();
            if (t == null) continue;
            if (!(t.getGpuTag() instanceof GpuBuffer gb)) continue;
            if (!gb.cpuStale) continue;

            runtime.downloadBuffer(gb.id, t.data);
            gb.cpuStale = false;
            gb.needsUpload = false;
        }
    }

    private void clearTensorGpuTags() {
        for (WeakReference<Tensor> ref : bufIdToTensor.values()) {
            Tensor t = ref.get();
            if (t != null) t.setGpuTag(null);
        }
    }

    private void releaseNativeBuffers() {
        if (allocatedBufferIds.isEmpty()) return;
        int[] ids = allocatedBufferIds.stream().mapToInt(Integer::intValue).toArray();
        runtime.releaseBuffers(ids, ids.length);
    }

    private void releaseNativeBuffers(List<Integer> buffers) {
        if (buffers.isEmpty()) return;
        int[] ids = buffers.stream().mapToInt(Integer::intValue).toArray();
        runtime.releaseBuffers(ids, ids.length);
    }

    private void resetGraphState() {
        bufIdToTensor.clear();
        pendingAllocs.clear();
        pendingUploadIds.clear();
        pendingUploadData.clear();
        allocatedBufferIds.clear();
        cmdPos = 0;
        opCount = 0;
    }

    private static int nextBufferId() {
        int id = NEXT_BUFFER_ID.getAndIncrement();
        if (id < 0) throw new IllegalStateException("GPU buffer id space exhausted");
        return id;
    }
}
