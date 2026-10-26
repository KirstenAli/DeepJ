package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.GpuBuffer;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorTestStorage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.*;

class MetalStorageCopyTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(DType.class)
    void ndCopiesAreLazyAndIdentityGradientsRemainFloat32(DType dtype) {
        Tensor source = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, dtype, 2, 1, 3).requiresGrad(true);
        Tensor copy = source.copy();
        assertFalse(TensorTestStorage.hasHostStorage(copy));
        assertEquals(dtype, copy.dtype());
        assertArrayEquals(source.shape(), copy.shape());
        assertArrayEquals(source.toFloatArray(), copy.toFloatArray());
        copy.backward(Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, source.shape()));
        assertEquals(DType.FLOAT32, source.grad().dtype());
        assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6}, source.grad().toFloatArray());
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void deviceCopyPreservesAllRawCompactBitsWithoutDownloadingItsInput(DType dtype) {
        GpuBuffer input = metal.graph.newOutputBuffer(1, 65_536, dtype);
        Tensor owner = metal.graph.createOutputTensor(input);
        metal.graph.flush();
        short[] expected = compactPatterns(), actual = new short[expected.length];
        MetalNative.nativeUploadBuffer16(input.id, expected);
        Tensor copy = metal.copy(owner);
        assertFalse(TensorTestStorage.hasHostStorage(owner));
        assertFalse(TensorTestStorage.hasHostStorage(copy));
        metal.graph.flush();
        MetalNative.nativeDownloadBuffer16(((GpuBuffer) copy.getGpuTag()).id, actual);
        assertArrayEquals(expected, actual);
        assertFalse(TensorTestStorage.hasHostStorage(owner));
        assertFalse(TensorTestStorage.hasHostStorage(copy));
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void oddOffsetsAndLengthsCopyExactlyWithoutBlitAlignmentAssumptions(DType dtype) {
        Tensor source = Tensor.from(new float[]{0, 1, 2, 3, 4, 5, 6}, dtype);
        GpuBuffer input = metal.graph.ensureStorageBuffer(source);
        GpuBuffer output = metal.graph.newZeroBuffer(1, 9, dtype);
        Tensor result = metal.graph.createOutputTensor(output, 3, 3);
        metal.graph.recordStorageCopy(input, output, 1, 3, 5);
        assertArrayEquals(new float[]{0, 0, 0, 1, 2, 3, 4, 5, 0}, result.toFloatArray());
        Tensor offset = source.slice(0, 1, 6, 1);
        assertArrayEquals(new float[]{1, 2, 3, 4, 5}, offset.copy().toFloatArray());
    }

    @Test
    void nativeCopyRejectsInvalidRangesBeforeDispatching() {
        Tensor first = new Tensor(1, 4), second = new Tensor(1, 4);
        int input = metal.graph.ensureGpuBuffer(first).id, output = metal.graph.ensureGpuBuffer(second).id;
        metal.graph.flush();
        invalid(input, output, -1, 0, 1);
        invalid(input, output, 0, -1, 1);
        invalid(input, output, 0, 0, 0);
        invalid(input, output, 0, 0, -1);
        invalid(input, output, 1, 0, 4);
        invalid(input, output, 0, 1, 4);
        invalid(input, input, 0, 0, 1);
        invalid(input, output, Integer.MAX_VALUE, 0, Integer.MAX_VALUE);
    }

    @Test
    void floatCopiesPreserveSignedZerosSubnormalsInfinitiesAndNaNPayloads() {
        int[] bits = {0, 0x80000000, 1, 0x007fffff, 0x7f800000, 0xff800000, 0x7f801234, 0xffc01234};
        GpuBuffer input = metal.graph.newOutputBuffer(1, bits.length);
        Tensor source = metal.graph.createOutputTensor(input, 2, 2, 2);
        metal.graph.flush();
        float[] expected = new float[bits.length], actual = new float[bits.length];
        for (int index = 0; index < bits.length; index++) { expected[index] = Float.intBitsToFloat(bits[index]); }
        MetalNative.nativeUploadBuffer(input.id, expected);
        Tensor copy = source.copy();
        metal.graph.flush();
        MetalNative.nativeDownloadBuffer(((GpuBuffer) copy.getGpuTag()).id, actual);
        for (int index = 0; index < bits.length; index++) {
            assertEquals(bits[index], Float.floatToRawIntBits(actual[index]));
        }
        assertFalse(TensorTestStorage.hasHostStorage(source));
    }

    @Test
    void copiesStayOrderedBetweenExistingComputeKernelsWithoutHostMaterialization() {
        Tensor source = Tensor.from2D(new float[][]{{1, -2, 3}});
        Tensor intermediate = source.neg();
        Tensor copy = intermediate.copy();
        Tensor result = copy.neg();
        assertFalse(TensorTestStorage.hasHostStorage(intermediate));
        assertFalse(TensorTestStorage.hasHostStorage(copy));
        assertArrayEquals(new float[]{1, -2, 3}, result.toFloatArray());
        assertFalse(TensorTestStorage.hasHostStorage(intermediate));
        assertFalse(TensorTestStorage.hasHostStorage(copy));
    }

    private static void invalid(int input, int output, int source, int target, int count) {
        int[] commands = {io.github.kirstenali.deepj.tensor.ComputeGraph.OP_STORAGE_COPY,
                input, output, source, target, count};
        assertThrows(RuntimeException.class, () -> MetalNative.nativeFlushOps(commands, commands.length));
    }

    private static short[] compactPatterns() {
        short[] values = new short[65_536];
        for (int index = 0; index < values.length; index++) { values[index] = (short) index; }
        return values;
    }
}
