package io.github.kirstenali.deepj.tensor;

import java.lang.ref.WeakReference;
import java.util.*;

abstract class GraphCore extends GraphOpMetadata {

    final GpuRuntime runtime;

    int[] cmdStream = new int[4096];
    int cmdPos = 0;
    int opCount = 0;
    private int operationStart;

    final List<int[]> pendingAllocs = new ArrayList<>();
    final List<int[]> pendingUploadIds = new ArrayList<>();
    final List<float[]> pendingUploadData = new ArrayList<>();
    final Map<Integer, WeakReference<Tensor>> bufIdToTensor = new HashMap<>();
    final Set<Integer> allocatedBufferIds = new HashSet<>();
    final Set<Integer> backwardBufferIds = new HashSet<>();
    final Map<Integer, GpuBuffer> managedBuffers = new HashMap<>();

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

    void uploadPendingData() {
        for (int i = 0; i < pendingUploadIds.size(); i++) {
            int[] upload = pendingUploadIds.get(i);
            float[] data = pendingUploadData.get(i);
            if (data == null) runtime.zeroBuffer(upload[0], upload[1]);
            else runtime.uploadBuffer(upload[0], data);
        }
        pendingUploadIds.clear();
        pendingUploadData.clear();
    }

    public GpuBuffer ensureGpuBuffer(Tensor t) {
        if (t.getGpuTag() instanceof GpuBuffer existing) {
            existing.requireLive();
            if (existing.owner != null && existing.owner != this) {
                existing.owner.materialize(t);
                t.setGpuTag(null);
                return createAndUploadInputBuffer(t);
            }
            return reuseExistingInputBuffer(t, existing);
        }

        return createAndUploadInputBuffer(t);
    }

    GpuBuffer reuseExistingInputBuffer(Tensor t, GpuBuffer existing) {
        existing.owner = (GraphExecution) this;
        trackTensorBinding(existing.id, t);
        if (existing.needsUpload) {
            scheduleUpload(existing.id, TensorAdapters.packF32(t));
            existing.needsUpload = false;
            existing.cpuStale = false;
        }
        return existing;
    }

    GpuBuffer createAndUploadInputBuffer(Tensor t) {
        GpuBuffer buf = newBuffer(t.rows, t.cols, true);
        scheduleUpload(buf.id, TensorAdapters.packF32(t));
        buf.needsUpload = false;
        buf.cpuStale = false;

        t.setGpuTag(buf);
        trackTensorBinding(buf.id, t);
        return buf;
    }

    void trackTensorBinding(int id, Tensor t) {
        bufIdToTensor.put(id, new WeakReference<>(t));
    }

    public GpuBuffer newOutputBuffer(int rows, int cols) {
        GpuBuffer buf = newBuffer(rows, cols, false);
        BackwardBuffers.track(buf);
        return buf;
    }

    /** Queues explicit zero initialization before any operations consume this buffer. */
    public GpuBuffer newZeroBuffer(int rows, int cols) {
        TensorStorage.checkedSize(rows, cols);
        GpuBuffer buffer = newOutputBuffer(rows, cols);
        pendingUploadIds.add(new int[]{buffer.id, buffer.floatCount()});
        pendingUploadData.add(null);
        return buffer;
    }

    public Tensor createOutputTensor(GpuBuffer buf) {
        Tensor t = Tensor.deviceOnly(buf.rows, buf.cols);
        bindTensorToBuffer(t, buf);
        return t;
    }

    public Tensor copyToDevice(Tensor source) {
        TensorStorage.requireSource(source);
        GpuBuffer buffer = newBuffer(source.rows, source.cols, false);
        scheduleUpload(buffer.id, TensorAdapters.packF32(source));
        return createOutputTensor(buffer);
    }

    public void markUpdated(Tensor tensor) {
        if (!(tensor.getGpuTag() instanceof GpuBuffer buffer)) {
            throw new IllegalArgumentException("Updated tensor must have a GPU buffer");
        }
        buffer.requireLive();
        if (buffer.owner != this) throw new IllegalArgumentException("GPU buffer belongs to another graph");
        buffer.cpuStale = true;
        buffer.needsUpload = false;
        TensorStorage.discardStaleHostStorage(tensor);
    }

    public void bindTensorToBuffer(Tensor t, GpuBuffer buf) {
        buf.requireLive();
        if (buf.owner != null && buf.owner != this) {
            throw new IllegalArgumentException("GPU buffers belong to their creating graph");
        }
        buf.owner = (GraphExecution) this;
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
        operationStart = cmdPos;
    }

    void emitBuffer(GpuBuffer buffer) {
        try {
            buffer.requireLive();
        } catch (RuntimeException failure) {
            cmdPos = operationStart;
            throw failure;
        }
        emitInt(buffer.id);
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

    private GpuBuffer newBuffer(int rows, int cols, boolean upload) {
        GpuBuffer buffer = new GpuBuffer(GpuBufferIds.SHARED.acquire(), rows, cols, upload);
        buffer.owner = (GraphExecution) this;
        managedBuffers.put(buffer.id, buffer);
        scheduleAlloc(buffer.id, buffer.floatCount());
        return buffer;
    }

}
