package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.*;

class MetalTypedTransferTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void nativeStorageUsesTwoBytesAndPreservesEveryRaw16BitPattern(DType dtype) {
        Tensor owner = new Tensor(dtype, 1, 65_536);
        var buffer = metal.graph.ensureStorageBuffer(owner);
        metal.graph.flush();
        assertEquals(131_072L, MetalNative.nativeBufferByteLength(buffer.id));
        short[] input = patterns(), actual = new short[input.length];
        MetalNative.nativeUploadBuffer16(buffer.id, input);
        MetalNative.nativeDownloadBuffer16(buffer.id, actual);
        assertArrayEquals(input, actual);
        MetalNative.nativeZeroTypedBuffer(buffer.id, input.length, dtype.storageCode());
        MetalNative.nativeDownloadBuffer16(buffer.id, actual);
        assertArrayEquals(new short[input.length], actual);
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void nativeValidationRejectsConflictingFormatsAndTransferSizes(DType dtype) {
        Tensor owner = new Tensor(dtype, 1, 4);
        int id = metal.graph.ensureStorageBuffer(owner).id;
        metal.graph.flush();
        assertThrows(RuntimeException.class, () -> MetalNative.nativeUploadBuffer(id, new float[2]));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeDownloadBuffer(id, new float[2]));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeUploadBuffer16(id, null));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeDownloadBuffer16(id, null));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeUploadBuffer16(id, new short[3]));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeDownloadBuffer16(-1, new short[4]));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeZeroBuffer(id, 2));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeZeroTypedBuffer(id, 4, 9));
        assertThrows(RuntimeException.class,
                () -> MetalNative.nativeAllocBuffers(new int[]{id}, new int[]{2}, 1));
        MetalNative.nativeAllocTypedBuffers(new int[]{id}, new int[]{4}, new int[]{dtype.storageCode()}, 1);
    }

    private static short[] patterns() {
        short[] values = new short[65_536];
        for (int index = 0; index < values.length; index++) { values[index] = (short) index; }
        return values;
    }
}
