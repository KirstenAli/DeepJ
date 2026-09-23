package io.github.kirstenali.deepj.tensor;

import java.lang.ref.WeakReference;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

abstract class GraphExecution extends GraphAdvancedRecordingOps {
    GraphExecution(GpuRuntime runtime) {
        super(runtime);
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

    void allocatePendingBuffers() {
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

    void markAllocatedBuffers() {
        for (var ref : bufIdToTensor.values()) {
            Tensor t = ref.get();
            if (t != null && t.getGpuTag() instanceof GpuBuffer gb) {
                gb.allocatedOnGpu = true;
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
            allocatedBufferIds.remove(id);
        }

        int[] ids = orphanIds.stream().mapToInt(Integer::intValue).toArray();
        runtime.releaseBuffers(ids, ids.length);
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
        if (t == null) return true;
        return !(t.getGpuTag() instanceof GpuBuffer gb) || gb.id != id;
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

    List<Integer> temporaryBufferIds() {
        List<Integer> ids = new ArrayList<>();
        for (int id : allocatedBufferIds) {
            WeakReference<Tensor> reference = bufIdToTensor.get(id);
            Tensor tensor = reference == null ? null : reference.get();
            if (tensor == null || !tensor.retainsDeviceBuffer()) ids.add(id);
        }
        return ids;
    }

    void clearTensorGpuTags(List<Integer> ids) {
        for (int id : ids) {
            WeakReference<Tensor> reference = bufIdToTensor.remove(id);
            Tensor tensor = reference == null ? null : reference.get();
            if (ownsBuffer(tensor, id)) tensor.setGpuTag(null);
        }
    }

    static boolean ownsBuffer(Tensor tensor, int id) {
        return tensor != null && tensor.getGpuTag() instanceof GpuBuffer buffer
                && buffer.id == id;
    }

    void materializeTrackedTensors() {
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

    void clearTensorGpuTags() {
        for (WeakReference<Tensor> ref : bufIdToTensor.values()) {
            Tensor t = ref.get();
            if (t != null) t.setGpuTag(null);
        }
    }

    void releaseNativeBuffers() {
        if (allocatedBufferIds.isEmpty()) return;
        int[] ids = allocatedBufferIds.stream().mapToInt(Integer::intValue).toArray();
        runtime.releaseBuffers(ids, ids.length);
    }

    void releaseNativeBuffers(List<Integer> buffers) {
        if (buffers.isEmpty()) return;
        int[] ids = buffers.stream().mapToInt(Integer::intValue).toArray();
        runtime.releaseBuffers(ids, ids.length);
    }

    void resetGraphState() {
        bufIdToTensor.clear();
        pendingAllocs.clear();
        pendingUploadIds.clear();
        pendingUploadData.clear();
        allocatedBufferIds.clear();
        cmdPos = 0;
        opCount = 0;
    }

}
