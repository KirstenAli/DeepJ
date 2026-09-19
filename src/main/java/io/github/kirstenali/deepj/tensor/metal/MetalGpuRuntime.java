package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.GpuRuntime;

import java.util.Objects;

final class MetalGpuRuntime implements GpuRuntime {

    @Override
    public void allocBuffers(int[] ids, int[] sizes, int count) {
        requireCount(count, Objects.requireNonNull(ids), Objects.requireNonNull(sizes));
        for (int i = 0; i < count; i++) {
            if (ids[i] < 0 || sizes[i] <= 0) throw new IllegalArgumentException("Invalid GPU buffer allocation");
        }
        MetalNative.nativeAllocBuffers(ids, sizes, count);
    }

    @Override
    public void uploadBuffer(int bufId, float[] data) {
        MetalNative.nativeUploadBuffer(requireBufferId(bufId), Objects.requireNonNull(data));
    }

    @Override
    public void downloadBuffer(int bufId, float[] out) {
        MetalNative.nativeDownloadBuffer(requireBufferId(bufId), Objects.requireNonNull(out));
    }

    @Override
    public void releaseBuffers(int[] ids, int count) {
        Objects.requireNonNull(ids);
        if (count < 0 || count > ids.length) throw new IllegalArgumentException("Invalid release count");
        MetalNative.nativeReleaseBuffers(ids, count);
    }

    @Override
    public void flushOps(int[] cmdStream, int cmdStreamLength) {
        Objects.requireNonNull(cmdStream);
        if (cmdStreamLength < 0 || cmdStreamLength > cmdStream.length) {
            throw new IllegalArgumentException("Invalid command stream length");
        }
        MetalNative.nativeFlushOps(cmdStream, cmdStreamLength);
    }

    private static void requireCount(int count, int[] ids, int[] sizes) {
        if (count < 0 || count > ids.length || count > sizes.length) {
            throw new IllegalArgumentException("Invalid allocation count");
        }
    }

    private static int requireBufferId(int id) {
        if (id < 0) throw new IllegalArgumentException("Invalid GPU buffer id: " + id);
        return id;
    }
}
