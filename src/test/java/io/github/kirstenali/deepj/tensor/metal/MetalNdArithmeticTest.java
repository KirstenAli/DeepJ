package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.CompactFloatReference;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.TensorTestStorage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class MetalNdArithmeticTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(DType.class)
    void binaryOperationsPreserveRankZeroThroughSixAndRoundOnlyTheirOutput(DType dtype) {
        for (int rank = 0; rank <= 6; rank++) {
            int[] shape = new int[rank];
            Arrays.fill(shape, 2);
            float[] values = new float[1 << rank];
            for (int index = 0; index < values.length; index++) { values[index] = (index - 7) * 0.125f; }
            Tensor left = Tensor.from(values, dtype, shape), right = cpu.addScalar(cpu.zeros(dtype, shape), 2);
            compare(left, right);
        }
    }

    @Test
    void trailingBroadcastsPermutationsAndMixedDtypesUseLogicalAddresses() {
        for (DType leftType : DType.values()) {
            for (DType rightType : DType.values()) { compareBroadcast(leftType, rightType); }
        }
    }

    private void compareBroadcast(DType first, DType second) {
        Tensor left = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, first, 2, 3).permute(1, 0).unsqueeze(0);
        Tensor right = Tensor.from(new float[]{2, 4, 8, 16}, second, 2, 1, 2);
        compare(left, right);
        Tensor scalar = Tensor.from(new float[]{2}, second, new int[0]);
        compare(left.slice(1, 0, 3, 2), scalar);
    }

    private void compare(Tensor left, Tensor right) {
        for (int operation = 0; operation < 4; operation++) {
            Tensor expected = binary(cpu, left, right, operation), actual = binary(metal, left, right, operation);
            assertFalse(TensorTestStorage.hasHostStorage(actual));
            assertEquals(expected.dtype(), actual.dtype());
            assertArrayEquals(expected.shape(), actual.shape());
            assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
        }
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void randomizedBroadcastOutputsMatchIndependentCoordinatesAndStoredValueArithmetic(DType dtype) {
        Random random = new Random(20261026);
        float[] values = new float[21];
        for (int index = 0; index < values.length; index++) { values[index] = random.nextFloat() * 4 - 2; }
        Tensor left = Tensor.from(values, dtype, 7, 1, 3).permute(2, 1, 0);
        Tensor right = Tensor.from(new float[]{0.25f, 0.5f, 1, 2, 4}, dtype, 1, 5, 1);
        for (int operation = 0; operation < 4; operation++) {
            Tensor result = binary(metal, left, right, operation);
            assertArrayEquals(new int[]{3, 5, 7}, result.shape());
            float[] expected = independent(left, right, dtype, operation);
            assertArrayEquals(expected, result.toFloatArray());
        }
    }

    private static float[] independent(Tensor left, Tensor right, DType dtype, int operation) {
        float[] result = new float[105];
        for (int index = 0; index < result.length; index++) {
            float a = left.get(index / 35, 0, index % 7), b = right.get(0, index / 7 % 5, 0);
            float value = calculate(a, b, operation);
            result[index] = quantize(value, dtype);
        }
        return result;
    }

    private static float quantize(float value, DType dtype) {
        return switch (dtype) {
            case FLOAT32 -> value;
            case FLOAT16 -> CompactFloatReference.half(Float.floatToFloat16(value) & 0xffff);
            case BFLOAT16 -> Float.intBitsToFloat((CompactFloatReference.bfloat(value) & 0xffff) << 16);
        };
    }

    private static float calculate(float left, float right, int operation) {
        return switch (operation) {
            case 0 -> left + right;
            case 1 -> left - right;
            case 2 -> left * right;
            default -> left / right;
        };
    }

    private static Tensor binary(TensorBackend backend, Tensor left, Tensor right, int operation) {
        return switch (operation) {
            case 0 -> backend.add(left, right);
            case 1 -> backend.subtract(left, right);
            case 2 -> backend.multiply(left, right);
            default -> backend.divide(left, right);
        };
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void generalGpuArithmeticDoesNotMaterializeInputsAndKeepsRealStorageWidths(DType dtype) {
        Tensor source = metal.addScalar(metal.zeros(2, 3), 2).to(dtype);
        Tensor view = source.permute(1, 0);
        Tensor output = metal.add(metal.multiply(view, view), metal.neg(view));
        assertFalse(TensorTestStorage.hasHostStorage(source));
        assertFalse(TensorTestStorage.hasHostStorage(output));
        assertEquals(dtype.storageBytes(6), output.storageBytes());
        assertArrayEquals(new float[]{2, 2, 2, 2, 2, 2}, output.toFloatArray());
        assertFalse(TensorTestStorage.hasHostStorage(source));
    }

    @Test
    void incompatibleAndEmptyBroadcastsAreValidatedWithoutAllocatingValues() {
        Tensor empty = metal.multiply(new Tensor(DType.FLOAT16, 2, 0, 3), new Tensor(DType.BFLOAT16, 1, 3));
        assertArrayEquals(new int[]{2, 0, 3}, empty.shape());
        assertEquals(DType.FLOAT32, empty.dtype());
        assertEquals(0, empty.numel());
        assertNull(empty.getGpuTag());
        assertThrows(IllegalArgumentException.class, () -> metal.add(new Tensor(2, 3), new Tensor(4)));
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void repeatedOperandsAccumulateIndependentFloat32AdjointsOnGpu(DType dtype) {
        Tensor source = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, dtype, 1, 2, 3).requiresGrad(true);
        Tensor output = source.multiply(source).add(source);
        output.backward(Tensor.from(new float[]{1, 1, 1, 1, 1, 1}, 1, 2, 3));
        assertEquals(DType.FLOAT32, source.grad().dtype());
        assertArrayEquals(source.shape(), source.grad().shape());
        assertArrayEquals(new float[]{3, 5, 7, 9, 11, 13}, source.grad().toFloatArray());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void typedZerosAndZeroPowerGradientsKeepLogicalShapeWithoutHostArrays(DType dtype) {
        Tensor zero = metal.zeros(dtype, 2, 1, 3);
        assertEquals(dtype, zero.dtype());
        assertFalse(TensorTestStorage.hasHostStorage(zero));
        assertArrayEquals(new float[6], zero.toFloatArray());
        Tensor source = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, dtype, 2, 1, 3).requiresGrad(true);
        source.pow(0).backward(Tensor.from(new float[]{1, 1, 1, 1, 1, 1}, 2, 1, 3));
        assertArrayEquals(new float[6], source.grad().toFloatArray());
        assertArrayEquals(new int[]{2, 1, 3}, source.grad().shape());
        assertNull(metal.zeros(dtype, 2, 0, 3).getGpuTag());
    }
}
