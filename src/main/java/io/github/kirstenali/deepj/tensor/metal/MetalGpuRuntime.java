package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.GpuRuntime;
import io.github.kirstenali.deepj.tensor.GpuAllocations;
import io.github.kirstenali.deepj.tensor.DType;

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
    public void allocTypedBuffers(int[] ids, int[] elements, DType[] dtypes, int count) {
        GpuAllocations.requireValid(ids, elements, dtypes, count);
        int[] codes = new int[count];
        for (int index = 0; index < count; index++) {
            codes[index] = dtypes[index].storageCode();
        }
        MetalNative.nativeAllocTypedBuffers(ids, elements, codes, count);
    }

    @Override
    public void uploadBuffer(int bufId, float[] data) {
        MetalNative.nativeUploadBuffer(requireBufferId(bufId), Objects.requireNonNull(data));
    }

    @Override
    public void uploadBuffer16(int bufId, short[] data) {
        MetalNative.nativeUploadBuffer16(requireBufferId(bufId), Objects.requireNonNull(data));
    }

    @Override
    public void zeroBuffer(int bufId, int floatCount) {
        if (floatCount < 1) throw new IllegalArgumentException("Invalid zero buffer size");
        MetalNative.nativeZeroBuffer(requireBufferId(bufId), floatCount);
    }

    @Override
    public void zeroBuffer(int bufId, int elements, DType dtype) {
        Objects.requireNonNull(dtype);
        if (elements < 1) throw new IllegalArgumentException("Invalid zero buffer size");
        MetalNative.nativeZeroTypedBuffer(requireBufferId(bufId), elements, dtype.storageCode());
    }

    @Override
    public void downloadBuffer(int bufId, float[] out) {
        MetalNative.nativeDownloadBuffer(requireBufferId(bufId), Objects.requireNonNull(out));
    }

    @Override
    public void downloadBuffer16(int bufId, short[] out) {
        MetalNative.nativeDownloadBuffer16(requireBufferId(bufId), Objects.requireNonNull(out));
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
