package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.GpuRuntime;

/**
 * Apple Metal implementation of {@link GpuRuntime}.
 * Delegates every call to the JNI functions in {@link MetalNative}.
 */
final class MetalGpuRuntime implements GpuRuntime {

    @Override
    public void allocBuffers(int[] ids, int[] sizes, int count) {
        MetalNative.nativeAllocBuffers(ids, sizes, count);
    }

    @Override
    public void uploadBuffer(int bufId, float[] data) {
        MetalNative.nativeUploadBuffer(bufId, data);
    }

    @Override
    public void downloadBuffer(int bufId, float[] out) {
        MetalNative.nativeDownloadBuffer(bufId, out);
    }

    @Override
    public void releaseBuffers(int[] ids, int count) {
        MetalNative.nativeReleaseBuffers(ids, count);
    }

    @Override
    public void flushOps(int[] cmdStream, int cmdStreamLength) {
        MetalNative.nativeFlushOps(cmdStream, cmdStreamLength);
    }
}

