package io.github.kirstenali.deepj.tensor;

import java.lang.ref.WeakReference;
import java.util.*;

/**
 * Collects GPU operations lazily and flushes them as a single command buffer.
 *
 * <p>Operations are recorded as a flat {@code int[]} command stream. Buffer IDs reference
 * persistent GPU-side buffers managed by a {@link GpuRuntime}. Data stays GPU-resident
 * between ops -- only uploaded at graph entry and downloaded on materialization.
 *
 * <p>This class is <b>backend-agnostic</b>: Metal, CUDA, Vulkan, etc. are all supported
 * by supplying the appropriate {@link GpuRuntime} implementation.
 */
public final class ComputeGraph {

    // Op codes (must match native side)
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

    private record OpMeta(int stride, int[] bufferArgOffsets) {}

    // Single source of truth for op stride and which encoded arg slots are buffer IDs.
    private static final OpMeta[] OP_METADATA = buildOpMetadata();

    private final GpuRuntime runtime;

    private int nextBufId = 0;

    private int[] cmdStream = new int[4096];
    private int cmdPos = 0;
    private int opCount = 0;

    private final List<int[]> pendingAllocs = new ArrayList<>();
    private final List<int[]> pendingUploadIds = new ArrayList<>();
    private final List<float[]> pendingUploadData = new ArrayList<>();
    private final Map<Integer, WeakReference<Tensor>> bufIdToTensor = new HashMap<>();

    /**
     * Create a ComputeGraph backed by the given GPU runtime.
     *
     * @param runtime the native driver abstraction (Metal, CUDA, etc.)
     */
    public ComputeGraph(GpuRuntime runtime) {
        this.runtime = Objects.requireNonNull(runtime, "runtime");
    }

    private static OpMeta[] buildOpMetadata() {
        OpMeta[] meta = new OpMeta[OP_SCATTER_ADD_ROWS + 1];

        // Unary: [op, in, out, n]
        registerMeta(meta, OP_SQRT, 4, 1, 2);
        registerMeta(meta, OP_NEG, 4, 1, 2);
        registerMeta(meta, OP_EXP, 4, 1, 2);
        registerMeta(meta, OP_LOG, 4, 1, 2);
        registerMeta(meta, OP_TANH, 4, 1, 2);
        registerMeta(meta, OP_SIGMOID, 4, 1, 2);
        registerMeta(meta, OP_RELU, 4, 1, 2);
        registerMeta(meta, OP_GELU, 4, 1, 2);

        // Binary elementwise: [op, a, b, out, n]
        registerMeta(meta, OP_ADD, 5, 1, 2, 3);
        registerMeta(meta, OP_SUBTRACT, 5, 1, 2, 3);
        registerMeta(meta, OP_MULTIPLY, 5, 1, 2, 3);
        registerMeta(meta, OP_DIVIDE, 5, 1, 2, 3);
        registerMeta(meta, OP_RELU_BACKWARD, 5, 1, 2, 3);
        registerMeta(meta, OP_GELU_BACKWARD, 5, 1, 2, 3);

        // Scalar and reduction style: [op, in, out, ..., ...]
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

        // 3-input ops with shape args
        registerMeta(meta, OP_SOFTMAX_BACKWARD, 6, 1, 2, 3);
        registerMeta(meta, OP_ADD_ROW_VECTOR, 6, 1, 2, 3);
        registerMeta(meta, OP_ADD_BROADCAST_COLS, 6, 1, 2, 3);
        registerMeta(meta, OP_SUBTRACT_BROADCAST_COLS, 6, 1, 2, 3);
        registerMeta(meta, OP_DIVIDE_BROADCAST_COLS, 6, 1, 2, 3);
        registerMeta(meta, OP_MULTIPLY_BROADCAST_ROWS, 6, 1, 2, 3);
        registerMeta(meta, OP_MULTIPLY_BROADCAST_COLS, 6, 1, 2, 3);

        // Matmul and layernorm backward
        registerMeta(meta, OP_MATMUL, 7, 1, 2, 3);
        registerMeta(meta, OP_LAYERNORM_BACKWARD, 7, 1, 2, 3, 4);

        // AdamW in-place update
        registerMeta(meta, OP_ADAMW_UPDATE, 13, 1, 2, 3, 4);

        return meta;
    }

    private static void registerMeta(OpMeta[] meta, int op, int stride, int... bufferArgOffsets) {
        meta[op] = new OpMeta(stride, bufferArgOffsets);
    }

    // -- Scheduling helpers --------------------------------------------------

    /** Queue a GPU buffer allocation for the next flush. */
    private void scheduleAlloc(int id, int floatCount) {
        pendingAllocs.add(new int[]{id, floatCount});
    }

    /** Queue a CPU-to-GPU data upload for the next flush. */
    private void scheduleUpload(int id, float[] data) {
        pendingUploadIds.add(new int[]{id});
        pendingUploadData.add(data);
    }

    // -- Buffer management ---------------------------------------------------

    /**
     * Ensure a tensor has a GpuBuffer. If it already has one (from a previous op),
     * reuse it. Otherwise allocate a new buffer and schedule upload of its CPU data.
     */
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
        int id = nextBufId++;
        GpuBuffer buf = new GpuBuffer(id, t.rows, t.cols, true);
        buf.allocatedOnGpu = false;

        scheduleAlloc(id, buf.floatCount());
        scheduleUpload(id, TensorAdapters.packF32(t));

        t.setGpuTag(buf);
        trackTensorBinding(id, t);
        return buf;
    }

    private void trackTensorBinding(int id, Tensor t) {
        bufIdToTensor.put(id, new WeakReference<>(t));
    }

    /**
     * Allocate a new output buffer (result of a GPU op). Not yet allocated on native side.
     */
    public GpuBuffer newOutputBuffer(int rows, int cols) {
        int id = nextBufId++;
        GpuBuffer buf = new GpuBuffer(id, rows, cols, false);
        buf.cpuStale = true;
        buf.allocatedOnGpu = false;
        scheduleAlloc(id, buf.floatCount());
        return buf;
    }

    /**
     * Create a Tensor backed by a GpuBuffer. The data[] buffer is allocated but stale.
     */
    public Tensor createOutputTensor(GpuBuffer buf) {
        Tensor t = new Tensor(buf.rows, buf.cols);
        bindTensorToBuffer(t, buf);
        return t;
    }

    /**
     * Rebind an existing tensor to a GPU buffer and track ownership for lifecycle management.
     */
    public void bindTensorToBuffer(Tensor t, GpuBuffer buf) {
        t.setGpuTag(buf);
        trackTensorBinding(buf.id, t);
    }

    // -- Op recording --------------------------------------------------------

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

    /** Record a binary element-wise op: [opCode, aId, bId, outId, n] */
    public void recordBinary(int opCode, GpuBuffer a, GpuBuffer b, GpuBuffer out) {
        beginOp(5);
        emitInt(opCode);
        emitInt(a.id);
        emitInt(b.id);
        emitInt(out.id);
        emitInt(out.floatCount());
        endOp();
    }

    /** Record matmul: [OP_MATMUL, aId, bId, outId, m, n, k] */
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

    /** Record a unary op: [opCode, inId, outId, n] */
    public void recordUnary(int opCode, GpuBuffer in, GpuBuffer out) {
        beginOp(4);
        emitInt(opCode);
        emitInt(in.id);
        emitInt(out.id);
        emitInt(out.floatCount());
        endOp();
    }

    /** Record scalar multiply: [OP_MULTIPLY_SCALAR, inId, outId, scalarBits, n] */
    public void recordMultiplyScalar(GpuBuffer in, GpuBuffer out, float scalar) {
        beginOp(5);
        emitInt(OP_MULTIPLY_SCALAR);
        emitInt(in.id);
        emitInt(out.id);
        emitFloatBits(scalar);
        emitInt(out.floatCount());
        endOp();
    }

    /** Record scalar add/divide: [opCode, inId, outId, scalarBits, n] */
    public void recordScalarUnary(int opCode, GpuBuffer in, GpuBuffer out, float scalar) {
        beginOp(5);
        emitInt(opCode);
        emitInt(in.id);
        emitInt(out.id);
        emitFloatBits(scalar);
        emitInt(out.floatCount());
        endOp();
    }

    /** Record pow: [OP_POW, inId, outId, exponentBits, n] */
    public void recordPow(GpuBuffer in, GpuBuffer out, float exponent) {
        beginOp(5);
        emitInt(OP_POW);
        emitInt(in.id);
        emitInt(out.id);
        emitFloatBits(exponent);
        emitInt(out.floatCount());
        endOp();
    }

    /** Record clamp: [OP_CLAMP, inId, outId, minBits, maxBits, n] */
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

    /** Record scatter-add-rows: [OP_SCATTER_ADD_ROWS, targetId, indicesId, gradId, targetRows, targetCols, nIdx] */
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

    /** Record transpose: [OP_TRANSPOSE, inId, outId, rows, cols] */
    public void recordTranspose(GpuBuffer in, GpuBuffer out, int rows, int cols) {
        beginOp(5);
        emitInt(OP_TRANSPOSE);
        emitInt(in.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    /** Record row broadcast: [opCode, aId, rowVecId, outId, rows, cols] */
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

    /** Record col broadcast: [opCode, aId, colVecId, outId, rows, cols] */
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

    /** Record row/col reduction: [opCode, inId, outId, rows, cols] */
    public void recordReduction(int opCode, GpuBuffer in, GpuBuffer out, int rows, int cols) {
        beginOp(5);
        emitInt(opCode);
        emitInt(in.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    /** Record softmax rows: [OP_SOFTMAX_ROWS, inId, outId, rows, cols] */
    public void recordSoftmaxRows(GpuBuffer in, GpuBuffer out, int rows, int cols) {
        beginOp(5);
        emitInt(OP_SOFTMAX_ROWS);
        emitInt(in.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    /** Record softmax backward: [OP_SOFTMAX_BACKWARD, gradId, softmaxId, outId, rows, cols] */
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

    /** Record layer norm backward: [OP_LAYERNORM_BACKWARD, dXHatId, xHatId, stdId, outId, rows, cols] */
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

    /**
     * Record in-place AdamW update:
     * [OP_ADAMW_UPDATE, wId, gId, mtId, vtId,
     *  lrBits, beta1Bits, beta2Bits, epsBits, weightDecayBits, bc1Bits, bc2Bits, n]
     */
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

    // -- Flush: execute everything in one command buffer ---------------------

    /**
     * Flush all recorded ops to the GPU as a single command buffer.
     * After flush, GPU buffers hold computed results; CPU data is stale.
     */
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

    /** Batch-allocate all GPU buffers that were requested since the last flush. */
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

    /** Mark every tracked GpuBuffer as allocated on the GPU side. */
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
        }

        int[] ids = orphanIds.stream().mapToInt(Integer::intValue).toArray();
        runtime.releaseBuffers(ids, ids.length);
    }

    /** Collect IDs of buffers whose owning Tensor no longer references them. */
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

    /** Returns true when {@code t} is gone or no longer points at buffer {@code id}. */
    private boolean isOrphanedBuffer(int id, Tensor t) {
        if (t == null) return true;
        return !(t.getGpuTag() instanceof GpuBuffer gb) || gb.id != id;
    }

    // -- Op stream traversal helpers -----------------------------------------

    /**
     * Returns the total number of ints consumed by {@code op} in the command stream,
     * or {@code -1} for an unknown op.
     */
    private static int getOpStride(int op) {
        if (op < 0 || op >= OP_METADATA.length) return -1;
        OpMeta meta = OP_METADATA[op];
        return meta == null ? -1 : meta.stride();
    }

    /**
     * Returns {@code true} if any buffer-ID slot at {@code pos} in the command stream
     * contains {@code bufferId}.
     */
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
            if (stride < 0) return true; // unknown op: be conservative
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

    /** Upload all CPU tensor data that is queued for transfer to the GPU. */
    private void uploadPendingData() {
        if (pendingUploadIds.isEmpty()) return;

        for (int i = 0; i < pendingUploadIds.size(); i++) {
            runtime.uploadBuffer(pendingUploadIds.get(i)[0], pendingUploadData.get(i));
        }
        pendingUploadIds.clear();
        pendingUploadData.clear();
    }

    /** Execute all recorded ops as one command buffer, then reset the op stream. */
    private void executePendingOps() {
        if (opCount > 0) {
            runtime.flushOps(cmdStream, cmdPos);
        }
        cmdPos  = 0;
        opCount = 0;
    }

    /**
     * Materialize a tensor: flush pending ops if needed, then download GPU data to CPU.
     */
    public void materialize(Tensor t) {
        if (!(t.getGpuTag() instanceof GpuBuffer buf)) return;
        if (!buf.cpuStale) return;

        flush();

        float[] flat = new float[buf.floatCount()];
        runtime.downloadBuffer(buf.id, flat);
        TensorAdapters.unpackF32Into(flat, t);
        buf.cpuStale = false;
    }

    /**
     * Release all GPU buffers and reset the graph completely.
     */
    public void releaseAll() {
        materializeTrackedTensors();
        clearTensorGpuTags();
        releaseNativeBuffers();
        resetGraphState();
    }

    /**
     * Before dropping GPU buffers, pull any stale tracked tensors back to CPU so
     * periodic release does not discard GPU-only updates (e.g., optimizer steps).
     */
    private void materializeTrackedTensors() {
        flush();
        for (WeakReference<Tensor> ref : bufIdToTensor.values()) {
            Tensor t = ref.get();
            if (t == null) continue;
            if (!(t.getGpuTag() instanceof GpuBuffer gb)) continue;
            if (!gb.cpuStale) continue;

            float[] flat = new float[gb.floatCount()];
            runtime.downloadBuffer(gb.id, flat);
            TensorAdapters.unpackF32Into(flat, t);
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
        if (!bufIdToTensor.isEmpty()) {
            int[] ids = bufIdToTensor.keySet().stream().mapToInt(Integer::intValue).toArray();
            runtime.releaseBuffers(ids, ids.length);
        }
    }

    private void resetGraphState() {
        bufIdToTensor.clear();
        pendingAllocs.clear();
        pendingUploadIds.clear();
        pendingUploadData.clear();
        cmdPos = 0;
        opCount = 0;
        nextBufId = 0;
    }
}

