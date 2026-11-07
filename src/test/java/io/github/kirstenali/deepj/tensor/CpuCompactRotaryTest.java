package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class CpuCompactRotaryTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void bothPairingConventionsPreserveScalarRoundingAndOffsets(DType dtype) {
        for (int width : new int[]{2, 128, 2048}) {
            Tensor input = Tensor.random(dtype, new Random(19), 65, width * 2).slice(0, 1, 65);
            Tensor strided = input.slice(1, 0, width * 2, 2);
            Tensor dense = new Tensor(strided);
            Tensor cos = Tensor.random(65, width / 2, new Random(20)).slice(0, 1, 65);
            Tensor sin = Tensor.random(65, width / 2, new Random(21)).slice(0, 1, 65);
            for (int flags = 0; flags < 4; flags++) { compare(strided, dense, cos, sin, flags); }
        }
    }

    private static void compare(Tensor strided, Tensor dense, Tensor cosine, Tensor sine, int flags) {
        boolean inverse = (flags & 1) != 0, halfSplit = (flags & 2) != 0;
        Tensor expected = CpuNdPosition.rotary(strided, cosine, sine, 64, inverse, halfSplit);
        Tensor actual = CpuNdPosition.rotary(dense, cosine, sine, 64, inverse, halfSplit);
        assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
    }
}
