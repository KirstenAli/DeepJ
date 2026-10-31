package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.GpuBuffer;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorTestStorage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class MetalLayoutCopyTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void stridedCopyPreservesAllRaw16BitPatternsIncludingNaNPayloads(DType dtype) {
        short[] bits = new short[65_536];
        for (int index = 0; index < bits.length; index++) { bits[index] = (short) index; }
        Tensor source = TensorTestStorage.from16Bits(bits, dtype, 256, 256);
        Tensor output = source.permute(1, 0).copy();
        assertFalse(TensorTestStorage.hasHostStorage(output));
        short[] actual = TensorTestStorage.bits16(output);
        for (int index = 0; index < actual.length; index++) {
            assertEquals(bits[(index % 256) * 256 + index / 256], actual[index]);
        }
        assertArrayEquals(new int[]{256, 256}, output.shape());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void offsetSlicesPermutationsAndBroadcastViewsCopyInLogicalOrder(DType dtype) {
        Tensor source = Tensor.from(new float[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, dtype, 3, 4);
        Tensor view = source.slice(0, 1, 3, 1).slice(1, 0, 4, 2).permute(1, 0);
        assertArrayEquals(new float[]{4, 8, 6, 10}, view.copy().toFloatArray());
        Tensor expanded = source.select(0, 1).unsqueeze(0).expand(3, 4);
        assertArrayEquals(new float[]{4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7}, expanded.copy().toFloatArray());
        assertEquals(dtype, expanded.copy().dtype());
    }

    @Test
    void layoutMetadataSupportsRanksBeyondTheMetalInlineArgumentLimit() {
        int[] shape = new int[3000];
        Arrays.fill(shape, 1);
        Tensor source = Tensor.from(new float[]{1.5f}, shape);
        Tensor output = source.to(DType.FLOAT16);
        assertEquals(3000, output.rank());
        assertEquals(1.5f, output.item());
        assertEquals(2, output.storageBytes());
    }

    @Test
    void layoutCastsCopiesAndInverseViewGradientsStayOnMetal() {
        Tensor source = Tensor.from(new float[]{1.5f, 2.25f, 3, 4, 5, 6}, 2, 3).requiresGrad(true);
        Tensor output = source.permute(1, 0).to(DType.FLOAT16).to(DType.FLOAT32).copy();
        assertFalse(TensorTestStorage.hasHostStorage(output));
        output.backward(Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, 3, 2));
        assertArrayEquals(new float[]{1, 3, 5, 2, 4, 6}, source.grad().toFloatArray());
        assertEquals(DType.FLOAT32, source.grad().dtype());
        assertArrayEquals(new int[]{2, 3}, source.grad().shape());
        GpuBuffer gradientBuffer = (GpuBuffer) source.grad().getGpuTag();
        assertEquals(2, gradientBuffer.rows);
        assertEquals(3, gradientBuffer.cols);
    }

    @Test
    void copyingAStridedGpuResultDoesNotAllocateAHostShadowOfTheInput() {
        Tensor source = metal.zeros(2, 3);
        Tensor view = source.permute(1, 0);
        Tensor output = view.copy();
        assertFalse(TensorTestStorage.hasHostStorage(source));
        assertFalse(TensorTestStorage.hasHostStorage(view));
        assertArrayEquals(new float[6], output.toFloatArray());
        assertFalse(TensorTestStorage.hasHostStorage(source));
        assertFalse(TensorTestStorage.hasHostStorage(view));
    }

    @Test
    void numericCastsDoNotMaterializeGpuSourcesOrIntermediateCompactValues() {
        Tensor source = metal.addScalar(metal.zeros(2, 3), 1.25f);
        Tensor encoded = source.permute(1, 0).to(DType.FLOAT16);
        Tensor decoded = encoded.to(DType.FLOAT32);
        assertFalse(TensorTestStorage.hasHostStorage(source));
        assertFalse(TensorTestStorage.hasHostStorage(encoded));
        assertArrayEquals(new float[]{1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f}, decoded.toFloatArray());
        assertFalse(TensorTestStorage.hasHostStorage(source));
        assertFalse(TensorTestStorage.hasHostStorage(encoded));
    }
}
