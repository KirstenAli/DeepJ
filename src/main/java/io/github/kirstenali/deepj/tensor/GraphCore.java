package io.github.kirstenali.deepj.tensor;

import java.lang.ref.WeakReference;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

abstract class GraphCore extends GraphOpMetadata {

    final GpuRuntime runtime;

    static final AtomicInteger NEXT_BUFFER_ID = new AtomicInteger();

    int[] cmdStream = new int[4096];
    int cmdPos = 0;
    int opCount = 0;

    final List<int[]> pendingAllocs = new ArrayList<>();
    final List<int[]> pendingUploadIds = new ArrayList<>();
    final List<float[]> pendingUploadData = new ArrayList<>();
    final Map<Integer, WeakReference<Tensor>> bufIdToTensor = new HashMap<>();
    final Set<Integer> allocatedBufferIds = new HashSet<>();

    GraphCore(GpuRuntime runtime) {
        this.runtime = Objects.requireNonNull(runtime, "runtime");
    }

    void scheduleAlloc(int id, int floatCount) {
        pendingAllocs.add(new int[]{id, floatCount});
        allocatedBufferIds.add(id);
    }

    void scheduleUpload(int id, float[] data) {
        pendingUploadIds.add(new int[]{id});
        pendingUploadData.add(data);
    }

    public GpuBuffer ensureGpuBuffer(Tensor t) {
        if (t.getGpuTag() instanceof GpuBuffer existing) {
            return reuseExistingInputBuffer(t, existing);
        }

        return createAndUploadInputBuffer(t);
    }

    GpuBuffer reuseExistingInputBuffer(Tensor t, GpuBuffer existing) {
        trackTensorBinding(existing.id, t);
        if (existing.needsUpload) {
            scheduleUpload(existing.id, TensorAdapters.packF32(t));
            existing.needsUpload = false;
            existing.cpuStale = false;
        }
        return existing;
    }

    GpuBuffer createAndUploadInputBuffer(Tensor t) {
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

    void trackTensorBinding(int id, Tensor t) {
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

    void ensureCapacity(int needed) {
        if (cmdPos + needed > cmdStream.length) {
            cmdStream = Arrays.copyOf(cmdStream, Math.max(cmdStream.length * 2, cmdPos + needed));
        }
    }

    void beginOp(int encodedInts) {
        ensureCapacity(encodedInts);
    }

    void emitInt(int value) {
        cmdStream[cmdPos++] = value;
    }

    void emitFloatBits(float value) {
        emitInt(Float.floatToRawIntBits(value));
    }

    void endOp() {
        opCount++;
    }

    static int nextBufferId() {
        int id = NEXT_BUFFER_ID.getAndIncrement();
        if (id < 0) throw new IllegalStateException("GPU buffer id space exhausted");
        return id;
    }

}
