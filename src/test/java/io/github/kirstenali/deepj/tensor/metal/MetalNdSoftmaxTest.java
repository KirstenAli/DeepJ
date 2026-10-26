package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorTestStorage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.*;

class MetalNdSoftmaxTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(DType.class)
    void arbitraryAxesAndStridedInputsNormalizeWithFloat32Intermediates(DType dtype) {
        Tensor input = Tensor.from(new float[]{-2, -1, 0, 1, 2, 3, -3, -2, -1, 0, 1, 2}, dtype, 2, 2, 3);
        for (int axis = 0; axis < input.rank(); axis++) { compare(input, axis); }
        compare(input.permute(2, 0, 1).slice(0, 0, 3, 2), -1);
        compare(input.select(0, 0).unsqueeze(0).expand(3, 2, 3), 0);
    }

    private void compare(Tensor input, int axis) {
        Tensor expected = cpu.softmaxAxis(input, axis), actual = metal.softmaxAxis(input, axis);
        assertFalse(TensorTestStorage.hasHostStorage(actual));
        assertEquals(input.dtype(), actual.dtype());
        assertArrayEquals(input.shape(), actual.shape());
        assertArrayEquals(expected.toFloatArray(), actual.toFloatArray(), 3e-6f);
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void nonfiniteRowsAndEmptyAxesMatchTheStableCpuContract(DType dtype) {
        Tensor input = Tensor.from(new float[]{Float.POSITIVE_INFINITY, 1, 2,
                Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY,
                Float.NaN, 0, 1}, dtype, 3, 3);
        for (float value : input.softmax(1).toFloatArray()) { assertTrue(Float.isNaN(value)); }
        Tensor mixed = Tensor.from(new float[]{Float.NEGATIVE_INFINITY, 0}, dtype);
        assertArrayEquals(new float[]{0, 1}, mixed.softmax(0).toFloatArray());
        Tensor empty = new Tensor(dtype, 2, 0, 3);
        assertEquals(0, empty.softmax(1).numel());
        assertNull(empty.getGpuTag());
        assertThrows(IndexOutOfBoundsException.class, () -> Tensor.from(new float[]{1}, new int[0]).softmax(0));
    }

    @Test
    void lazyInputsAndIntermediateNormalizationDoNotCreateHostShadows() {
        Tensor source = metal.addScalar(metal.zeros(3, 4), 2).to(DType.FLOAT16);
        Tensor output = source.unsqueeze(0).softmax(1);
        assertArrayEquals(new float[]{0.33325195f, 0.33325195f, 0.33325195f, 0.33325195f,
                0.33325195f, 0.33325195f, 0.33325195f, 0.33325195f,
                0.33325195f, 0.33325195f, 0.33325195f, 0.33325195f}, output.toFloatArray());
        assertFalse(TensorTestStorage.hasHostStorage(source));
    }
}
