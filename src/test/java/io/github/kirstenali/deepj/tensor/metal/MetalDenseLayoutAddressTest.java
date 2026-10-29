package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorTestStorage;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.*;

class MetalDenseLayoutAddressTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(DType.class)
    void denseOffsetArithmeticAndBackwardUseLogicalAddresses(DType dtype) {
        Tensor source = Tensor.from(sequence(35), dtype, 5, 7).requiresGrad(true);
        Tensor result = source.slice(0, 1, 4).multiplyScalar(2);
        assertFalse(TensorTestStorage.hasHostStorage(result));
        float[] expected = new float[21];
        for (int index = 0; index < expected.length; index++) { expected[index] = (index + 7) * 2; }
        assertArrayEquals(expected, result.toFloatArray());
        result.backward(Tensor.ones(DType.FLOAT32, 3, 7));
        float[] gradient = new float[35];
        for (int index = 7; index < 28; index++) { gradient[index] = 2; }
        assertArrayEquals(gradient, source.grad().toFloatArray());
        assertEquals(DType.FLOAT32, source.grad().dtype());
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void denseOffsetCopyPreservesEveryRawCompactPattern(DType dtype) {
        short[] bits = new short[258 * 256];
        for (int index = 0; index < bits.length; index++) { bits[index] = (short) (index - 256); }
        Tensor source = TensorTestStorage.from16Bits(bits, dtype, 258, 256);
        Tensor result = source.slice(0, 1, 257).copy();
        assertFalse(TensorTestStorage.hasHostStorage(result));
        short[] actual = TensorTestStorage.bits16(result);
        assertEquals(65_536, actual.length);
        for (int index = 0; index < actual.length; index++) { assertEquals((short) index, actual[index]); }
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void transposedAndRowStridedMatricesKeepTheirGeneralAddressing(DType dtype) {
        Tensor source = Tensor.from(sequence(35), dtype, 5, 7);
        Tensor view = source.slice(0, 0, 5, 2).permute(1, 0);
        float[] expected = new float[21];
        for (int index = 0; index < expected.length; index++) {
            expected[index] = 2 * ((index % 3) * 14 + index / 3);
        }
        assertArrayEquals(expected, view.multiplyScalar(2).toFloatArray());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void zeroStrideBroadcastsDoNotTakeTheDenseAddressPath(DType dtype) {
        Tensor source = Tensor.from(new float[]{1, 2, 3}, dtype, 1, 3);
        Tensor expanded = source.expand(4, 3);
        assertArrayEquals(new float[]{2, 4, 6, 2, 4, 6, 2, 4, 6, 2, 4, 6},
                expanded.multiplyScalar(2).toFloatArray());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void otherRanksAndEmptyMatricesRemainValid(DType dtype) {
        Tensor values = Tensor.from(sequence(12), dtype, 2, 2, 3);
        assertArrayEquals(new float[]{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22},
                values.multiplyScalar(2).toFloatArray());
        assertEquals(4, Tensor.from(new float[]{2}, dtype, new int[0]).multiplyScalar(2).item());
        assertEquals(0, Tensor.zeros(dtype, 0, 3).multiplyScalar(2).numel());
    }

    private static float[] sequence(int length) {
        float[] result = new float[length];
        for (int index = 0; index < length; index++) { result[index] = index; }
        return result;
    }
}
