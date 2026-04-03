package io.github.kirstenali.deepj.tensor;

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

    private final GpuRuntime runtime;

    private int nextBufId = 0;

    private int[] cmdStream = new int[4096];
    private int cmdPos = 0;
    private int opCount = 0;

    private final List<int[]> pendingAllocs = new ArrayList<>();
    private final List<int[]> pendingUploadIds = new ArrayList<>();
    private final List<float[]> pendingUploadData = new ArrayList<>();
    private final Map<Integer, Tensor> bufIdToTensor = new HashMap<>();

    /**
     * Create a ComputeGraph backed by the given GPU runtime.
     *
     * @param runtime the native driver abstraction (Metal, CUDA, etc.)
     */
    public ComputeGraph(GpuRuntime runtime) {
        this.runtime = Objects.requireNonNull(runtime, "runtime");
    }

    // -- Buffer management ---------------------------------------------------

    /**
     * Ensure a tensor has a GpuBuffer. If it already has one (from a previous op),
     * reuse it. Otherwise allocate a new buffer and schedule upload of its CPU data.
     */
    public GpuBuffer ensureGpuBuffer(Tensor t) {
        if (t.getGpuTag() instanceof GpuBuffer existing) {
            if (existing.needsUpload) {
                pendingUploadIds.add(new int[]{existing.id});
                pendingUploadData.add(TensorAdapters.packF32(t));
                existing.needsUpload = false;
                existing.cpuStale = false;
            }
            return existing;
        }
        int id = nextBufId++;
        GpuBuffer buf = new GpuBuffer(id, t.rows, t.cols, true);
        buf.allocatedOnGpu = false;

        pendingAllocs.add(new int[]{id, buf.floatCount()});
        pendingUploadIds.add(new int[]{id});
        pendingUploadData.add(TensorAdapters.packF32(t));

        t.setGpuTag(buf);
        bufIdToTensor.put(id, t);
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

        pendingAllocs.add(new int[]{id, buf.floatCount()});
        return buf;
    }

    /**
     * Create a Tensor backed by a GpuBuffer. The data[][] is allocated but stale.
     */
    public Tensor createOutputTensor(GpuBuffer buf) {
        Tensor t = new Tensor(buf.rows, buf.cols);
        t.setGpuTag(buf);
        bufIdToTensor.put(buf.id, t);
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

    public boolean isEmpty() { return opCount == 0; }

    // -- Flush: execute everything in one command buffer ---------------------

    /**
     * Flush all recorded ops to the GPU as a single command buffer.
     * After flush, GPU buffers hold computed results; CPU data is stale.
     */
    public void flush() {
        if (opCount == 0 && pendingAllocs.isEmpty()) return;

        allocatePendingBuffers();
        uploadPendingData();
        executePendingOps();
    }

    /** Batch-allocate all GPU buffers that were requested since the last flush. */
    private void allocatePendingBuffers() {
        if (pendingAllocs.isEmpty()) return;

        int[] ids   = new int[pendingAllocs.size()];
        int[] sizes = new int[pendingAllocs.size()];
        for (int i = 0; i < pendingAllocs.size(); i++) {
            ids[i]   = pendingAllocs.get(i)[0];
            sizes[i] = pendingAllocs.get(i)[1];
        }
        runtime.allocBuffers(ids, sizes, ids.length);

        for (var entry : bufIdToTensor.values()) {
            if (entry.getGpuTag() instanceof GpuBuffer gb) {
                gb.allocatedOnGpu = true;
            }
        }
        pendingAllocs.clear();
    }

    /** Upload all CPU tensor data that is queued for transfer to the GPU. */
    private void uploadPendingData() {
        if (pendingUploadIds.isEmpty()) return;

        for (int i = 0; i < pendingUploadIds.size(); i++) {
            int bufId    = pendingUploadIds.get(i)[0];
            float[] data = pendingUploadData.get(i);
            runtime.uploadBuffer(bufId, data);
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
        if (!bufIdToTensor.isEmpty()) {
            int[] ids = bufIdToTensor.keySet().stream().mapToInt(Integer::intValue).toArray();
            runtime.releaseBuffers(ids, ids.length);
        }
        bufIdToTensor.clear();
        pendingAllocs.clear();
        pendingUploadIds.clear();
        pendingUploadData.clear();
        cmdPos = 0;
        opCount = 0;
        nextBufId = 0;
    }
}

