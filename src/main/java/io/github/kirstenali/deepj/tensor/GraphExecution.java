package io.github.kirstenali.deepj.tensor;

import java.lang.ref.WeakReference;
import java.util.*;

abstract class GraphExecution extends GraphAttentionRecordingOps {
    GraphExecution(GpuRuntime runtime) {
        super(runtime);
    }
    public boolean isEmpty() { return opCount == 0; }

    public void flush() {
        if (opCount == 0 && pendingAllocs.isEmpty()) {
            releaseOrphanedBuffers();
            return;
        }

        GraphBufferReuse.Plan plan = GraphBufferReuse.prepare(this);
        allocatePendingBuffers(plan.allocations());
        uploadPendingData();
        executePendingOps(plan.commands());
        releaseOrphanedBuffers();
    }

    void allocatePendingBuffers() {
        allocatePendingBuffers(pendingAllocs);
    }

    private void allocatePendingBuffers(List<int[]> allocations) {
        if (allocations.isEmpty()) return;

        GpuAllocations.allocate(runtime, allocations);
        markAllocatedBuffers();
        pendingAllocs.clear();
    }

    void markAllocatedBuffers() {
        for (var entry : bufIdToMemory.entrySet()) {
            TensorMemory memory = entry.getValue().get();
            if (ownsBuffer(memory, entry.getKey())) {
                ((GpuBuffer) memory.gpuTag).allocatedOnGpu = true;
            }
        }
    }

    void releaseOrphanedBuffers() {
        if (bufIdToMemory.isEmpty()) return;

        List<Integer> orphanIds = collectOrphanIds();
        if (orphanIds.isEmpty()) return;

        for (int id : orphanIds) {
            removePendingForId(id);
            bufIdToMemory.remove(id);
        }

        int[] ids = orphanIds.stream().mapToInt(Integer::intValue).toArray();
        GraphBufferRelease.release(this, ids);
    }

    List<Integer> collectOrphanIds() {
        List<Integer> orphanIds = new ArrayList<>();
        for (var entry : bufIdToMemory.entrySet()) {
            int id = entry.getKey();
            TensorMemory memory = entry.getValue().get();
            if (isOrphanedBuffer(id, memory) && !isBufferReferencedByPendingOps(id)) {
                orphanIds.add(id);
            }
        }
        return orphanIds;
    }

    boolean isOrphanedBuffer(int id, TensorMemory memory) {
        if (backwardBufferIds.contains(id)) return true;
        return !ownsBuffer(memory, id);
    }

    void retireBackwardBuffer(GpuBuffer buffer) {
        if (managedBuffers.get(buffer.id) != buffer) return;
        var reference = bufIdToMemory.get(buffer.id);
        TensorMemory memory = reference == null ? null : reference.get();
        if (memory != null && !memory.canReleaseTemporary()) return;
        backwardBufferIds.add(buffer.id);
        if (ownsBuffer(memory, buffer.id)) memory.gpuTag = null;
        bufIdToMemory.putIfAbsent(buffer.id, new WeakReference<>(null));
    }

    static int getOpStride(int op) {
        if (op < 0 || op >= OP_METADATA.length) return -1;
        OpMeta meta = OP_METADATA[op];
        return meta == null ? -1 : meta.stride();
    }

    boolean opReferencesBuffer(int pos, int op, int bufferId) {
        if (op < 0 || op >= OP_METADATA.length) return false;
        OpMeta meta = OP_METADATA[op];
        if (meta == null) return false;

        for (int offset : meta.bufferArgOffsets()) {
            if (cmdStream[pos + offset] == bufferId) return true;
        }
        return false;
    }

    boolean isBufferReferencedByPendingOps(int bufferId) {
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

    void removePendingForId(int id) {
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

    void executePendingOps() {
        executePendingOps(cmdStream);
    }

    private void executePendingOps(int[] commands) {
        if (opCount > 0) {
            if (layouts.isEmpty()) runtime.flushOps(commands, cmdPos);
            else runtime.flushOps(commands, cmdPos, layouts.snapshot());
        }
        cmdPos  = 0;
        opCount = 0;
        layouts.clear();
    }

    public void materialize(Tensor t) {
        if (!(t.getGpuTag() instanceof GpuBuffer buf)) return;
        buf.requireLive();
        if (buf.owner != null && buf.owner != this) {
            buf.owner.materialize(t);
            return;
        }
        if (!buf.cpuStale) return;

        flush();

        GraphTransfers.download(runtime, buf, t.memory());
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
        backwardBufferIds.removeAll(ids);
    }

    List<Integer> temporaryBufferIds() {
        List<Integer> ids = new ArrayList<>();
        for (int id : allocatedBufferIds) {
            WeakReference<TensorMemory> reference = bufIdToMemory.get(id);
            TensorMemory memory = reference == null ? null : reference.get();
            if (memory == null || memory.canReleaseTemporary()) ids.add(id);
        }
        return ids;
    }

    void clearTensorGpuTags(List<Integer> ids) {
        for (int id : ids) {
            WeakReference<TensorMemory> reference = bufIdToMemory.remove(id);
            TensorMemory memory = reference == null ? null : reference.get();
            if (ownsBuffer(memory, id)) memory.gpuTag = null;
        }
    }

    boolean ownsBuffer(TensorMemory memory, int id) {
        return memory != null && memory.gpuTag instanceof GpuBuffer buffer
                && buffer.id == id && buffer.owner == this;
    }

    void materializeTrackedTensors() {
        flush();
        for (var entry : bufIdToMemory.entrySet()) {
            TensorMemory memory = entry.getValue().get();
            if (!ownsBuffer(memory, entry.getKey())) continue;
            GpuBuffer gb = (GpuBuffer) memory.gpuTag;
            if (!gb.cpuStale) continue;

            GraphTransfers.download(runtime, gb, memory);
        }
    }

    void clearTensorGpuTags() {
        for (var entry : bufIdToMemory.entrySet()) {
            TensorMemory memory = entry.getValue().get();
            if (ownsBuffer(memory, entry.getKey())) memory.gpuTag = null;
        }
    }

    void releaseNativeBuffers() {
        if (allocatedBufferIds.isEmpty()) return;
        int[] ids = allocatedBufferIds.stream().mapToInt(Integer::intValue).toArray();
        GraphBufferRelease.release(this, ids);
    }

    void releaseNativeBuffers(List<Integer> buffers) {
        if (buffers.isEmpty()) return;
        int[] ids = buffers.stream().mapToInt(Integer::intValue).toArray();
        GraphBufferRelease.release(this, ids);
    }

    void resetGraphState() {
        layouts.clear();
        bufIdToMemory.clear();
        pendingAllocs.clear();
        pendingUploadIds.clear();
        pendingUploadData.clear();
        allocatedBufferIds.clear();
        backwardBufferIds.clear();
        managedBuffers.clear();
        cmdPos = 0;
        opCount = 0;
    }

}
