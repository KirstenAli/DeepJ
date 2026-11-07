package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class TensorValuesCopyTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @EnumSource(DType.class)
    void copiesPreserveLogicalOrderAndRawBitsAcrossContiguousSuffixes(DType dtype) {
        Tensor source = Tensor.random(dtype, new Random(17), 3, 5, 7, 16);
        check(source.permute(0, 2, 1, 3));
        check(source.slice(0, 1, 3));
        check(source.slice(3, 1, 16, 2));
        check(source.select(0, 1).unsqueeze(0).expand(4, 5, 7, 16));
        check(source.slice(0, 0, 0));
        check(Tensor.from(new float[]{-0.0f, Float.NaN, Float.POSITIVE_INFINITY}, dtype, 1, 3).expand(7, 3));
    }

    private static void check(Tensor source) {
        Tensor copy = new Tensor(source);
        assertNotSame(source.memory(), copy.memory());
        assertArrayEquals(source.shape(), copy.shape());
        for (int index = 0; index < source.numel(); index++) {
            int address = source.layout().addressAt(index);
            if (source.dtype() == DType.FLOAT32) {
                assertEquals(Float.floatToRawIntBits(source.memory().floats()[address]),
                        Float.floatToRawIntBits(copy.memory().floats()[index]));
            } else assertEquals(source.memory().shorts()[address], copy.memory().shorts()[index]);
        }
    }
}
