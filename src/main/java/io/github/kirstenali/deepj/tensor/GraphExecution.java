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

        int count = allocations.size();
        int[] ids   = new int[count];
        int[] sizes = new int[count];
        for (int i = 0; i < count; i++) {
            ids[i]   = allocations.get(i)[0];
            sizes[i] = allocations.get(i)[1];
        }
        runtime.allocBuffers(ids, sizes, count);
        markAllocatedBuffers();
        pendingAllocs.clear();
    }

    void markAllocatedBuffers() {
        for (var entry : bufIdToTensor.entrySet()) {
            Tensor t = entry.getValue().get();
            if (ownsBuffer(t, entry.getKey())) {
                ((GpuBuffer) t.getGpuTag()).allocatedOnGpu = true;
            }
        }
    }

    void releaseOrphanedBuffers() {
        if (bufIdToTensor.isEmpty()) return;

        List<Integer> orphanIds = collectOrphanIds();
        if (orphanIds.isEmpty()) return;

        for (int id : orphanIds) {
            removePendingForId(id);
            bufIdToTensor.remove(id);
        }

        int[] ids = orphanIds.stream().mapToInt(Integer::intValue).toArray();
        GraphBufferRelease.release(this, ids);
    }

    List<Integer> collectOrphanIds() {
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

    boolean isOrphanedBuffer(int id, Tensor t) {
        if (backwardBufferIds.contains(id)) return true;
        if (t == null) return true;
        return !(t.getGpuTag() instanceof GpuBuffer gb) || gb.id != id;
    }

    void retireBackwardBuffer(GpuBuffer buffer) {
        if (managedBuffers.get(buffer.id) != buffer) return;
        var reference = bufIdToTensor.get(buffer.id);
        Tensor tensor = reference == null ? null : reference.get();
        if (tensor != null && !canReleaseTemporary(tensor)) return;
        backwardBufferIds.add(buffer.id);
        if (ownsBuffer(tensor, buffer.id)) tensor.setGpuTag(null);
        bufIdToTensor.putIfAbsent(buffer.id, new WeakReference<>(null));
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

    void uploadPendingData() {
        if (pendingUploadIds.isEmpty()) return;

        for (int i = 0; i < pendingUploadIds.size(); i++) {
            runtime.uploadBuffer(pendingUploadIds.get(i)[0], pendingUploadData.get(i));
        }
        pendingUploadIds.clear();
        pendingUploadData.clear();
    }

    void executePendingOps() {
        executePendingOps(cmdStream);
    }

    private void executePendingOps(int[] commands) {
        if (opCount > 0) {
            runtime.flushOps(commands, cmdPos);
        }
        cmdPos  = 0;
        opCount = 0;
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

        runtime.downloadBuffer(buf.id, t.hostStorage());
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
        backwardBufferIds.removeAll(ids);
    }

    List<Integer> temporaryBufferIds() {
        List<Integer> ids = new ArrayList<>();
        for (int id : allocatedBufferIds) {
            WeakReference<Tensor> reference = bufIdToTensor.get(id);
            Tensor tensor = reference == null ? null : reference.get();
            if (tensor == null || canReleaseTemporary(tensor)) ids.add(id);
        }
        return ids;
    }

    private static boolean canReleaseTemporary(Tensor tensor) {
        return !tensor.retainsDeviceBuffer() && !tensor.gradientState().savedForBackward();
    }

    void clearTensorGpuTags(List<Integer> ids) {
        for (int id : ids) {
            WeakReference<Tensor> reference = bufIdToTensor.remove(id);
            Tensor tensor = reference == null ? null : reference.get();
            if (ownsBuffer(tensor, id)) tensor.setGpuTag(null);
        }
    }

    boolean ownsBuffer(Tensor tensor, int id) {
        return tensor != null && tensor.getGpuTag() instanceof GpuBuffer buffer
                && buffer.id == id && buffer.owner == this;
    }

    void materializeTrackedTensors() {
        flush();
        for (var entry : bufIdToTensor.entrySet()) {
            Tensor t = entry.getValue().get();
            if (!ownsBuffer(t, entry.getKey())) continue;
            GpuBuffer gb = (GpuBuffer) t.getGpuTag();
            if (!gb.cpuStale) continue;

            runtime.downloadBuffer(gb.id, t.hostStorage());
            gb.cpuStale = false;
            gb.needsUpload = false;
        }
    }

    void clearTensorGpuTags() {
        for (var entry : bufIdToTensor.entrySet()) {
            Tensor t = entry.getValue().get();
            if (ownsBuffer(t, entry.getKey())) t.setGpuTag(null);
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
        bufIdToTensor.clear();
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
