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
    final List<Object> pendingUploadData = new ArrayList<>();
    final Map<Integer, WeakReference<TensorMemory>> bufIdToMemory = new HashMap<>();
    final Set<Integer> allocatedBufferIds = new HashSet<>();
    final Set<Integer> backwardBufferIds = new HashSet<>();
    final Map<Integer, GpuBuffer> managedBuffers = new HashMap<>();

    GraphCore(GpuRuntime runtime) {
        this.runtime = Objects.requireNonNull(runtime, "runtime");
    }

    void scheduleAlloc(GpuBuffer buffer) {
        int[] allocation = buffer.dtype == DType.FLOAT32
                ? new int[]{buffer.id, buffer.elementCount()}
                : new int[]{buffer.id, buffer.elementCount(), buffer.dtype.storageCode()};
        pendingAllocs.add(allocation);
        allocatedBufferIds.add(buffer.id);
    }

    void scheduleUpload(int id, Object data) {
        pendingUploadIds.add(new int[]{id});
        pendingUploadData.add(data);
    }

    void uploadPendingData() {
        for (int i = 0; i < pendingUploadIds.size(); i++) {
            GraphTransfers.upload(runtime, pendingUploadIds.get(i), pendingUploadData.get(i));
        }
        pendingUploadIds.clear();
        pendingUploadData.clear();
    }

    public GpuBuffer ensureGpuBuffer(Tensor t) {
        requireMatrixStorage(t);
        return ensureStorageBuffer(t);
    }

    /** Physical shared storage only; logical ND kernels must supply their own layouts. */
    public GpuBuffer ensureStorageBuffer(Tensor t) {
        TensorStorage.requireSource(t);
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
        requireStorageBinding(t, existing);
        existing.owner = (GraphExecution) this;
        trackTensorBinding(existing.id, t);
        if (existing.needsUpload) {
            if (((GraphExecution) this).isBufferReferencedByPendingOps(existing.id)) {
                ((GraphExecution) this).flush();
            }
            scheduleUpload(existing.id, GraphTransfers.snapshot(t));
            existing.needsUpload = false;
            existing.cpuStale = false;
        }
        return existing;
    }

    GpuBuffer createAndUploadInputBuffer(Tensor t) {
        if (t.memory().length() == 0) throw new IllegalArgumentException("Empty tensors need no GPU allocation");
        int rows = t.rank() == 2 && t.coversStorage() ? t.matrixRows() : 1;
        GpuBuffer buf = newBuffer(rows, t.memory().length() / rows, t.dtype(), true);
        scheduleUpload(buf.id, GraphTransfers.snapshot(t));
        buf.needsUpload = false;
        buf.cpuStale = false;

        t.setGpuTag(buf);
        trackTensorBinding(buf.id, t);
        return buf;
    }

    void trackTensorBinding(int id, Tensor t) {
        bufIdToMemory.put(id, new WeakReference<>(t.memory()));
    }

    public GpuBuffer newOutputBuffer(int rows, int cols) {
        return newOutputBuffer(rows, cols, DType.FLOAT32);
    }

    public GpuBuffer newOutputBuffer(int rows, int cols, DType dtype) {
        GpuBuffer buf = newBuffer(rows, cols, dtype, false);
        BackwardBuffers.track(buf);
        return buf;
    }

    /** Queues explicit zero initialization before any operations consume this buffer. */
    public GpuBuffer newZeroBuffer(int rows, int cols) {
        return newZeroBuffer(rows, cols, DType.FLOAT32);
    }

    public GpuBuffer newZeroBuffer(int rows, int cols, DType dtype) {
        GpuBuffer buffer = newOutputBuffer(rows, cols, dtype);
        int[] initialization = dtype == DType.FLOAT32 ? new int[]{buffer.id, buffer.elementCount()}
                : new int[]{buffer.id, buffer.elementCount(), dtype.storageCode()};
        pendingUploadIds.add(initialization);
        pendingUploadData.add(null);
        return buffer;
    }

    public Tensor createOutputTensor(GpuBuffer buf) {
        return createOutputTensor(buf, buf.rows, buf.cols);
    }

    public Tensor createOutputTensor(GpuBuffer buf, int... shape) {
        if (new TensorShape(shape).elements() != buf.elementCount()) {
            throw new IllegalArgumentException("Output shape must cover the GPU allocation");
        }
        Tensor t = Tensor.deviceOnly(buf.dtype, shape);
        bindTensorToBuffer(t, buf);
        return t;
    }

    public Tensor copyToDevice(Tensor source) {
        TensorStorage.requireSource(source);
        requireMatrixStorage(source);
        GpuBuffer buffer = newBuffer(source.matrixRows(), source.matrixCols(), DType.FLOAT32, false);
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
        requireStorageBinding(t, buf);
        buf.owner = (GraphExecution) this;
        t.setGpuTag(buf);
        trackTensorBinding(buf.id, t);
    }

    private void requireStorageBinding(Tensor tensor, GpuBuffer buffer) {
        if (tensor.memory().length() != buffer.elementCount() || tensor.dtype() != buffer.dtype) {
            throw new IllegalArgumentException("GPU allocation size and dtype must match physical tensor storage");
        }
        var reference = bufIdToMemory.get(buffer.id);
        TensorMemory bound = reference == null ? null : reference.get();
        if (bound != null && bound != tensor.memory()) {
            throw new IllegalArgumentException("A GPU allocation can only bind one shared tensor storage");
        }
    }

    private static void requireMatrixStorage(Tensor tensor) {
        if (tensor.dtype() != DType.FLOAT32 || tensor.rank() != 2 || !tensor.coversStorage()) {
            throw new UnsupportedOperationException(
                    "The matrix graph requires full contiguous FLOAT32 rank-two storage");
        }
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
            buffer.requireFloat32();
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

    private GpuBuffer newBuffer(int rows, int cols, DType dtype, boolean upload) {
        TensorStorage.checkedSize(rows, cols);
        Objects.requireNonNull(dtype, "dtype");
        GpuBuffer buffer = new GpuBuffer(GpuBufferIds.SHARED.acquire(), rows, cols, dtype, upload);
        buffer.owner = (GraphExecution) this;
        managedBuffers.put(buffer.id, buffer);
        scheduleAlloc(buffer);
        return buffer;
    }

}
