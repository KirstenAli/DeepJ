package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class CpuLastAxisSelectionTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @EnumSource(DType.class)
    void offsetPermutedSteppedAndBroadcastRowsMatchLogicalSelection(DType dtype) {
        Tensor input = input(dtype, 3, 7, 4, 16);
        check(input.slice(0, 1, 3).permute(0, 2, 1, 3), new int[]{15, 0, 15, 4});
        check(input.permute(0, 3, 2, 1), new int[]{6, 0, 6, 3});
        check(input.slice(3, 1, 16, 2), new int[]{7, 1, 7, 0});
        check(input(dtype, 2, 1).expand(2, 4), new int[]{3, 0, 3});
        check(input(dtype, 17).slice(0, 1, 17, 2), new int[]{7, 0, 7});
        check(input, new int[0]);
        check(new Tensor(dtype, 0, 3), new int[]{2, 0});
        check(new Tensor(dtype, 2, 0), new int[0]);
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void largeGatherMatchesEveryValueWithSerialAndParallelExecution(DType dtype) {
        boolean previous = DeepJExecutor.isParallelEnabled();
        try {
            Tensor input = input(dtype, 1, 64, 16, 128).permute(0, 2, 1, 3);
            int[] indices = new int[128];
            for (int i = 0; i < indices.length; i++) { indices[i] = i / 2 + i % 2 * 64; }
            DeepJExecutor.setParallelEnabled(false);
            check(input, indices);
            DeepJExecutor.setParallelEnabled(true);
            check(input, indices);
        } finally {
            DeepJExecutor.setParallelEnabled(previous);
        }
    }

    @Test
    void rawStorageCopiesPreserveSignedZerosAndNanPayloads() {
        Tensor input = new Tensor(2, 3);
        int[] bits = {0x80000000, 0x7fc01234, 0xff800000, 0x7f800000, 0xffc05678, 1};
        for (int i = 0; i < bits.length; i++) { input.memory().floats()[i] = Float.intBitsToFloat(bits[i]); }
        Tensor output = input.indexSelect(-1, 2, 0, 2, 1);
        int[] expected = {bits[2], bits[0], bits[2], bits[1], bits[5], bits[3], bits[5], bits[4]};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], Float.floatToRawIntBits(output.memory().floats()[i]));
        }
    }

    @Test
    void selectedOutputsOwnTheirStorageAndKeepRepeatedIndexGradients() {
        Tensor input = input(DType.FLOAT32, 2, 1, 3).requiresGrad(true);
        Tensor selected = input.indexSelect(-1, 2, 0, 2);
        selected.sumTensor().backward();
        assertArrayEquals(new float[]{1, 0, 2, 1, 0, 2}, input.grad().toFloatArray());
        assertNotSame(input.memory(), selected.memory());
        float first = input.get(0, 0, 2);
        try (var ignored = GradMode.noGrad()) { selected.setAt(42, 0, 0, 0); }
        assertEquals(first, input.get(0, 0, 2));
    }

    private static void check(Tensor input, int[] indices) {
        float[] values = input.toFloatArray();
        int width = input.size(-1), rows = width == 0 ? 0 : values.length / width;
        float[] expected = new float[rows * indices.length];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < indices.length; column++) {
                expected[row * indices.length + column] = values[row * width + indices[column]];
            }
        }
        Tensor output = input.indexSelect(-1, indices);
        int[] shape = input.shape();
        shape[shape.length - 1] = indices.length;
        assertArrayEquals(shape, output.shape());
        assertArrayEquals(expected, output.toFloatArray());
        assertArrayEquals(values, input.toFloatArray());
    }

    private static Tensor input(DType dtype, int... shape) {
        float[] values = new float[new TensorShape(shape).elements()];
        for (int i = 0; i < values.length; i++) { values[i] = (i % 43 - 21) * 0.0625f; }
        return Tensor.from(values, dtype, shape);
    }
}
