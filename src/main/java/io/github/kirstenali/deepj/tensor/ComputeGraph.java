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
            bufIdToTensor.put(existing.id, new WeakReference<>(t));
            if (existing.needsUpload) {
                scheduleUpload(existing.id, TensorAdapters.packF32(t));
                existing.needsUpload = false;
                existing.cpuStale = false;
            }
            return existing;
        }
        int id = nextBufId++;
        GpuBuffer buf = new GpuBuffer(id, t.rows, t.cols, true);
        buf.allocatedOnGpu = false;

        scheduleAlloc(id, buf.floatCount());
        scheduleUpload(id, TensorAdapters.packF32(t));

        t.setGpuTag(buf);
        bufIdToTensor.put(id, new WeakReference<>(t));
        return buf;
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
     * Create a Tensor backed by a GpuBuffer. The data[][] is allocated but stale.
     */
    public Tensor createOutputTensor(GpuBuffer buf) {
        Tensor t = new Tensor(buf.rows, buf.cols);
        t.setGpuTag(buf);
        bufIdToTensor.put(buf.id, new WeakReference<>(t));
        return t;
    }

    // -- Op recording --------------------------------------------------------

    private void ensureCapacity(int needed) {
        if (cmdPos + needed > cmdStream.length) {
            cmdStream = Arrays.copyOf(cmdStream, Math.max(cmdStream.length * 2, cmdPos + needed));
        }
    }

    /** Record a binary element-wise op: [opCode, aId, bId, outId, n] */
    public void recordBinary(int opCode, GpuBuffer a, GpuBuffer b, GpuBuffer out) {
        ensureCapacity(5);
        cmdStream[cmdPos++] = opCode;
        cmdStream[cmdPos++] = a.id;
        cmdStream[cmdPos++] = b.id;
        cmdStream[cmdPos++] = out.id;
        cmdStream[cmdPos++] = out.floatCount();
        opCount++;
    }

    /** Record matmul: [OP_MATMUL, aId, bId, outId, m, n, k] */
    public void recordMatmul(GpuBuffer a, GpuBuffer b, GpuBuffer out, int m, int n, int k) {
        ensureCapacity(7);
        cmdStream[cmdPos++] = OP_MATMUL;
        cmdStream[cmdPos++] = a.id;
        cmdStream[cmdPos++] = b.id;
        cmdStream[cmdPos++] = out.id;
        cmdStream[cmdPos++] = m;
        cmdStream[cmdPos++] = n;
        cmdStream[cmdPos++] = k;
        opCount++;
    }

    /** Record a unary op: [opCode, inId, outId, n] */
    public void recordUnary(int opCode, GpuBuffer in, GpuBuffer out) {
        ensureCapacity(4);
        cmdStream[cmdPos++] = opCode;
        cmdStream[cmdPos++] = in.id;
        cmdStream[cmdPos++] = out.id;
        cmdStream[cmdPos++] = out.floatCount();
        opCount++;
    }

    /** Record scalar multiply: [OP_MULTIPLY_SCALAR, inId, outId, scalarBits, n] */
    public void recordMultiplyScalar(GpuBuffer in, GpuBuffer out, float scalar) {
        ensureCapacity(5);
        cmdStream[cmdPos++] = OP_MULTIPLY_SCALAR;
        cmdStream[cmdPos++] = in.id;
        cmdStream[cmdPos++] = out.id;
        cmdStream[cmdPos++] = Float.floatToRawIntBits(scalar);
        cmdStream[cmdPos++] = out.floatCount();
        opCount++;
    }

    /** Record scalar add/divide: [opCode, inId, outId, scalarBits, n] */
    public void recordScalarUnary(int opCode, GpuBuffer in, GpuBuffer out, float scalar) {
        ensureCapacity(5);
        cmdStream[cmdPos++] = opCode;
        cmdStream[cmdPos++] = in.id;
        cmdStream[cmdPos++] = out.id;
        cmdStream[cmdPos++] = Float.floatToRawIntBits(scalar);
        cmdStream[cmdPos++] = out.floatCount();
        opCount++;
    }

    /** Record transpose: [OP_TRANSPOSE, inId, outId, rows, cols] */
    public void recordTranspose(GpuBuffer in, GpuBuffer out, int rows, int cols) {
        ensureCapacity(5);
        cmdStream[cmdPos++] = OP_TRANSPOSE;
        cmdStream[cmdPos++] = in.id;
        cmdStream[cmdPos++] = out.id;
        cmdStream[cmdPos++] = rows;
        cmdStream[cmdPos++] = cols;
        opCount++;
    }

    /** Record row broadcast: [opCode, aId, rowVecId, outId, rows, cols] */
    public void recordRowBroadcast(int opCode, GpuBuffer a, GpuBuffer rowVec, GpuBuffer out, int rows, int cols) {
        ensureCapacity(6);
        cmdStream[cmdPos++] = opCode;
        cmdStream[cmdPos++] = a.id;
        cmdStream[cmdPos++] = rowVec.id;
        cmdStream[cmdPos++] = out.id;
        cmdStream[cmdPos++] = rows;
        cmdStream[cmdPos++] = cols;
        opCount++;
    }

    /** Record col broadcast: [opCode, aId, colVecId, outId, rows, cols] */
    public void recordColBroadcast(int opCode, GpuBuffer a, GpuBuffer colVec, GpuBuffer out, int rows, int cols) {
        ensureCapacity(6);
        cmdStream[cmdPos++] = opCode;
        cmdStream[cmdPos++] = a.id;
        cmdStream[cmdPos++] = colVec.id;
        cmdStream[cmdPos++] = out.id;
        cmdStream[cmdPos++] = rows;
        cmdStream[cmdPos++] = cols;
        opCount++;
    }

    /** Record row/col reduction: [opCode, inId, outId, rows, cols] */
    public void recordReduction(int opCode, GpuBuffer in, GpuBuffer out, int rows, int cols) {
        ensureCapacity(5);
        cmdStream[cmdPos++] = opCode;
        cmdStream[cmdPos++] = in.id;
        cmdStream[cmdPos++] = out.id;
        cmdStream[cmdPos++] = rows;
        cmdStream[cmdPos++] = cols;
        opCount++;
    }

    /** Record softmax rows: [OP_SOFTMAX_ROWS, inId, outId, rows, cols] */
    public void recordSoftmaxRows(GpuBuffer in, GpuBuffer out, int rows, int cols) {
        ensureCapacity(5);
        cmdStream[cmdPos++] = OP_SOFTMAX_ROWS;
        cmdStream[cmdPos++] = in.id;
        cmdStream[cmdPos++] = out.id;
        cmdStream[cmdPos++] = rows;
        cmdStream[cmdPos++] = cols;
        opCount++;
    }

    /** Record softmax backward: [OP_SOFTMAX_BACKWARD, gradId, softmaxId, outId, rows, cols] */
    public void recordSoftmaxBackward(GpuBuffer gradOutput, GpuBuffer softmaxOut, GpuBuffer out, int rows, int cols) {
        ensureCapacity(6);
        cmdStream[cmdPos++] = OP_SOFTMAX_BACKWARD;
        cmdStream[cmdPos++] = gradOutput.id;
        cmdStream[cmdPos++] = softmaxOut.id;
        cmdStream[cmdPos++] = out.id;
        cmdStream[cmdPos++] = rows;
        cmdStream[cmdPos++] = cols;
        opCount++;
    }

    /** Record layer norm backward: [OP_LAYERNORM_BACKWARD, dXHatId, xHatId, stdId, outId, rows, cols] */
    public void recordLayerNormBackward(GpuBuffer dXHat, GpuBuffer xHat, GpuBuffer std, GpuBuffer out, int rows, int cols) {
        ensureCapacity(7);
        cmdStream[cmdPos++] = OP_LAYERNORM_BACKWARD;
        cmdStream[cmdPos++] = dXHat.id;
        cmdStream[cmdPos++] = xHat.id;
        cmdStream[cmdPos++] = std.id;
        cmdStream[cmdPos++] = out.id;
        cmdStream[cmdPos++] = rows;
        cmdStream[cmdPos++] = cols;
        opCount++;
    }

    /**
     * Record in-place AdamW update:
     * [OP_ADAMW_UPDATE, wId, gId, mtId, vtId,
     *  lrBits, beta1Bits, beta2Bits, epsBits, weightDecayBits, bc1Bits, bc2Bits, n]
     */
    public void recordAdamWUpdate(GpuBuffer w, GpuBuffer g, GpuBuffer mt, GpuBuffer vt,
                                  float lr, float beta1, float beta2, float eps,
                                  float weightDecay, float bc1, float bc2, int n) {
        ensureCapacity(13);
        cmdStream[cmdPos++] = OP_ADAMW_UPDATE;
        cmdStream[cmdPos++] = w.id;
        cmdStream[cmdPos++] = g.id;
        cmdStream[cmdPos++] = mt.id;
        cmdStream[cmdPos++] = vt.id;
        cmdStream[cmdPos++] = Float.floatToRawIntBits(lr);
        cmdStream[cmdPos++] = Float.floatToRawIntBits(beta1);
        cmdStream[cmdPos++] = Float.floatToRawIntBits(beta2);
        cmdStream[cmdPos++] = Float.floatToRawIntBits(eps);
        cmdStream[cmdPos++] = Float.floatToRawIntBits(weightDecay);
        cmdStream[cmdPos++] = Float.floatToRawIntBits(bc1);
        cmdStream[cmdPos++] = Float.floatToRawIntBits(bc2);
        cmdStream[cmdPos++] = n;
        opCount++;
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
        return switch (op) {
            case OP_SQRT, OP_NEG, OP_EXP, OP_LOG, OP_TANH, OP_SIGMOID, OP_RELU, OP_GELU -> 4;
            case OP_ADD, OP_SUBTRACT, OP_MULTIPLY, OP_DIVIDE,
                 OP_RELU_BACKWARD, OP_GELU_BACKWARD, OP_MULTIPLY_SCALAR, OP_SOFTMAX_ROWS,
                 OP_ADD_SCALAR, OP_DIVIDE_SCALAR, OP_TRANSPOSE,
                 OP_SUM_ROWS, OP_MEAN_ALONG_ROWS, OP_VARIANCE_ALONG_ROWS,
                 OP_SUM_ALONG_ROWS -> 5;
            case OP_SOFTMAX_BACKWARD,
                 OP_ADD_ROW_VECTOR, OP_ADD_BROADCAST_COLS, OP_SUBTRACT_BROADCAST_COLS,
                 OP_DIVIDE_BROADCAST_COLS, OP_MULTIPLY_BROADCAST_ROWS,
                 OP_MULTIPLY_BROADCAST_COLS -> 6;
            case OP_MATMUL, OP_LAYERNORM_BACKWARD -> 7;
            case OP_ADAMW_UPDATE -> 13;
            default -> -1;
        };
    }

    /**
     * Returns {@code true} if any buffer-ID slot at {@code pos} in the command stream
     * contains {@code bufferId}.
     */
    private boolean opReferencesBuffer(int pos, int op, int bufferId) {
        return switch (op) {
            case OP_SQRT, OP_NEG, OP_EXP, OP_LOG, OP_TANH, OP_SIGMOID,
                 OP_RELU, OP_GELU, OP_MULTIPLY_SCALAR, OP_SOFTMAX_ROWS,
                 OP_ADD_SCALAR, OP_DIVIDE_SCALAR, OP_TRANSPOSE,
                 OP_SUM_ROWS, OP_MEAN_ALONG_ROWS, OP_VARIANCE_ALONG_ROWS,
                 OP_SUM_ALONG_ROWS ->
                cmdStream[pos + 1] == bufferId || cmdStream[pos + 2] == bufferId;
            case OP_ADD, OP_SUBTRACT, OP_MULTIPLY, OP_DIVIDE,
                 OP_RELU_BACKWARD, OP_GELU_BACKWARD, OP_MATMUL, OP_SOFTMAX_BACKWARD,
                 OP_ADD_ROW_VECTOR, OP_ADD_BROADCAST_COLS, OP_SUBTRACT_BROADCAST_COLS,
                 OP_DIVIDE_BROADCAST_COLS, OP_MULTIPLY_BROADCAST_ROWS,
                 OP_MULTIPLY_BROADCAST_COLS ->
                cmdStream[pos + 1] == bufferId || cmdStream[pos + 2] == bufferId
                        || cmdStream[pos + 3] == bufferId;
            case OP_LAYERNORM_BACKWARD, OP_ADAMW_UPDATE ->
                cmdStream[pos + 1] == bufferId || cmdStream[pos + 2] == bufferId
                        || cmdStream[pos + 3] == bufferId || cmdStream[pos + 4] == bufferId;
            default -> false;
        };
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

