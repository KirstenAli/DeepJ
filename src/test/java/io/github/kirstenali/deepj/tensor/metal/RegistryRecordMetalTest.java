package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.GpuBuffer;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.lang.ref.Reference;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/** Buffer identity, dtype and retention must remain coherent across native ID reuse. */
class RegistryRecordMetalTest extends MetalDifferentialTestSupport {
    private static final int ELEMENTS = 17;

    @ParameterizedTest
    @EnumSource(DType.class)
    void releasedIdCanBeReusedAcrossEveryStorageType(DType starting) {
        Tensor tensor = new Tensor(starting, 1, ELEMENTS);
        GpuBuffer token = metal.graph.ensureStorageBuffer(tensor);
        metal.graph.flush();
        int id = token.id;
        try {
            DType[] types = DType.values();
            for (int round = 0; round < 12; round++) {
                DType dtype = types[(starting.ordinal() + round) % types.length];
                verifyReusedId(id, dtype, round);
            }
        } finally {
            // Restore the Java graph's original logical allocation before its normal cleanup.
            MetalNative.nativeReleaseBuffers(new int[]{id}, 1);
            allocate(id, starting, ELEMENTS);
            Reference.reachabilityFence(tensor);
        }
    }

    private static void verifyReusedId(int id, DType dtype, int round) {
        MetalNative.nativeReleaseBuffers(new int[]{id, id}, 2);
        assertThrows(RuntimeException.class, () -> MetalNative.nativeBufferByteLength(id));
        allocate(id, dtype, ELEMENTS);
        assertEquals((long) ELEMENTS * dtype.bytes(), MetalNative.nativeBufferByteLength(id));
        verifyTransfer(id, dtype, round);
        allocate(id, dtype, ELEMENTS);
        assertThrows(RuntimeException.class, () -> allocate(id, dtype, ELEMENTS + 1));
        DType other = dtype == DType.FLOAT32 ? DType.FLOAT16 : DType.FLOAT32;
        assertThrows(RuntimeException.class, () -> allocate(id, other, ELEMENTS));
        MetalNative.nativeZeroTypedBuffer(id, ELEMENTS, dtype.storageCode());
        verifyZero(id, dtype);
    }

    private static void allocate(int id, DType dtype, int elements) {
        MetalNative.nativeAllocTypedBuffers(new int[]{id}, new int[]{elements}, new int[]{dtype.storageCode()}, 1);
    }

    private static void verifyZero(int id, DType dtype) {
        if (dtype == DType.FLOAT32) {
            float[] output = new float[ELEMENTS];
            MetalNative.nativeDownloadBuffer(id, output);
            assertArrayEquals(new float[ELEMENTS], output);
        } else {
            short[] output = new short[ELEMENTS];
            MetalNative.nativeDownloadBuffer16(id, output);
            assertArrayEquals(new short[ELEMENTS], output);
        }
    }

    private static void verifyTransfer(int id, DType dtype, int round) {
        if (dtype == DType.FLOAT32) verifyFloatTransfer(id, round);
        else verifyCompactTransfer(id, round);
    }

    private static void verifyFloatTransfer(int id, int round) {
        float[] input = new float[ELEMENTS], output = new float[ELEMENTS];
        for (int index = 0; index < input.length; index++) {
            input[index] = Float.intBitsToFloat(0x3f800000 + round * ELEMENTS + index);
        }
        MetalNative.nativeUploadBuffer(id, input);
        MetalNative.nativeDownloadBuffer(id, output);
        for (int index = 0; index < input.length; index++) {
            assertEquals(Float.floatToRawIntBits(input[index]), Float.floatToRawIntBits(output[index]));
        }
        assertThrows(RuntimeException.class, () -> MetalNative.nativeDownloadBuffer16(id, new short[2 * ELEMENTS]));
    }

    private static void verifyCompactTransfer(int id, int round) {
        short[] input = new short[ELEMENTS], output = new short[ELEMENTS];
        for (int index = 0; index < input.length; index++) {
            input[index] = (short) (0x8000 + round * ELEMENTS + index);
        }
        MetalNative.nativeUploadBuffer16(id, input);
        MetalNative.nativeDownloadBuffer16(id, output);
        assertArrayEquals(input, output);
        assertThrows(RuntimeException.class, () -> MetalNative.nativeUploadBuffer(id, new float[ELEMENTS]));
    }
}
