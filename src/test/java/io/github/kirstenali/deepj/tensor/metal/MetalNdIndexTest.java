package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorTestStorage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.*;

class MetalNdIndexTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(DType.class)
    void repeatedSelectionAndUpdatesRespectStridesAndOuterSequences(DType dtype) {
        Tensor source = Tensor.from(sequence(), dtype, 2, 3, 2);
        Tensor selected = source.permute(2, 1, 0).indexSelect(1, 2, 0, 2);
        assertArrayEquals(new float[]{4, 10, 0, 6, 4, 10, 5, 11, 1, 7, 5, 11}, selected.toFloatArray());
        Tensor target = metal.zeros(dtype, 2, 3, 2);
        target.indexAddInPlace(1, new int[]{2, 0, 2}, source);
        assertArrayEquals(new float[]{2, 3, 0, 0, 4, 6, 8, 9, 0, 0, 16, 18}, target.toFloatArray());
    }

    @Test
    void compactUpdatesRoundOnlyAfterAllRepeatedContributionsAndMixedUpdatesStayFloat32() {
        Tensor half = Tensor.from(new float[]{2048}, DType.FLOAT16);
        half.indexAddInPlace(0, new int[]{0, 0}, Tensor.from(new float[]{1, 1}, DType.FLOAT16));
        assertEquals(2050, half.item());
        Tensor full = Tensor.from(new float[]{1});
        full.indexAddInPlace(0, new int[]{0, 0}, Tensor.from(new float[]{2, 3}, DType.BFLOAT16));
        assertEquals(6, full.item());
        assertEquals(DType.FLOAT32, full.dtype());
    }

    @Test
    void repeatedUpdatesKeepChronologicalFloat32AdditionOrder() {
        Tensor target = metal.zeros(DType.FLOAT32, 2);
        Tensor updates = Tensor.from(new float[]{1e20f, 2, -1e20f, 3, 4});
        target.indexAddInPlace(0, new int[]{0, 1, 0, 0, 1}, updates);
        assertArrayEquals(new float[]{3, 6}, target.toFloatArray());
        target.indexAddInPlace(0, new int[]{0, 0, 0}, Tensor.from(new float[]{1e20f, -1e20f, 5}));
        assertArrayEquals(new float[]{5, 6}, target.toFloatArray());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void aliasedAndPartialUpdatesPreserveIndependentInputsAndParentFences(DType dtype) {
        Tensor owner = Tensor.from(new float[]{9, 1, 2, 8}, dtype);
        Tensor target = owner.slice(0, 1, 3, 1);
        target.indexAddInPlace(0, new int[]{1, 0}, target);
        assertArrayEquals(new float[]{9, 3, 3, 8}, owner.toFloatArray());
        Tensor matrix = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, dtype, 2, 3);
        matrix.permute(1, 0).indexAddInPlace(1, new int[]{1, 0}, Tensor.from(new float[]{1, 2, 3, 4, 5, 6},
                dtype, 3, 2));
        assertArrayEquals(new float[]{3, 6, 9, 5, 8, 11}, matrix.toFloatArray());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void gpuOutputsAndScatterTargetsDoNotAllocateHostShadows(DType dtype) {
        Tensor source = metal.copy(Tensor.from(new float[]{1, 2, 3}, dtype));
        Tensor output = source.indexSelect(0, 2, 0, 2), target = metal.zeros(dtype, 4);
        target.indexAddInPlace(0, new int[]{1, 3, 1}, output);
        metal.graph.flush();
        assertFalse(TensorTestStorage.hasHostStorage(source));
        assertFalse(TensorTestStorage.hasHostStorage(output));
        assertFalse(TensorTestStorage.hasHostStorage(target));
        assertArrayEquals(new float[]{0, 6, 0, 1}, target.toFloatArray());
    }

    @Test
    void rawHalfPayloadsAndIndicesBeyondFloatIntegerPrecisionRemainExact() {
        short[] bits = {(short) 0x7c01, (short) 0x8000, (short) 0xff81};
        Tensor source = TensorTestStorage.from16Bits(bits, DType.FLOAT16, 3);
        Tensor output = source.indexSelect(0, 2, 1, 0, 2);
        assertArrayEquals(new short[]{bits[2], bits[1], bits[0], bits[2]}, TensorTestStorage.bits16(output));
        Tensor large = new Tensor(DType.FLOAT16, 16_777_218);
        large.setAt(5, 16_777_217);
        assertArrayEquals(new float[]{0, 5}, large.indexSelect(0, 16_777_216, 16_777_217).toFloatArray());
    }

    @Test
    void emptySelectionsAndEmptyUpdateDimensionsHaveDefinedShapes() {
        Tensor source = new Tensor(DType.BFLOAT16, 2, 0, 3);
        assertArrayEquals(new int[]{2, 0, 3}, source.indexSelect(1).shape());
        source.indexAddInPlace(0, new int[]{1}, new Tensor(DType.BFLOAT16, 1, 0, 3));
        assertEquals(0, source.storageBytes());
        assertNull(source.getGpuTag());
    }

    private static float[] sequence() {
        float[] values = new float[12];
        for (int index = 0; index < values.length; index++) { values[index] = index; }
        return values;
    }
}
