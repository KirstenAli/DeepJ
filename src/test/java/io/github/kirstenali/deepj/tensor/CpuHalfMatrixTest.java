package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CpuHalfMatrixTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test void halfProductsRespectOffsetsPaddingAndTransposes() {
        Tensor a = values(66, 259).slice(0, 1, 65).slice(1, 1, 258);
        Tensor b = values(259, 131).slice(0, 1, 258).slice(1, 1, 130);
        compare(a, b);
        compare(a, values(129, 257).transpose(-1, -2));
        compare(values(257, 64).transpose(-1, -2), b);
        compare(values(257, 64).transpose(-1, -2), values(129, 257).transpose(-1, -2));
    }

    @Test void halfMatricesAccumulateInFloat32BeforeRounding() {
        float[] input = new float[64 * 256];
        for (int row = 0; row < 64; row++) {
            input[row * 256] = 4096;
            input[row * 256 + 1] = 1;
            input[row * 256 + 2] = -4096;
        }
        Tensor actual = Tensor.from(input, DType.FLOAT16, 64, 256)
                .matmul(Tensor.ones(DType.FLOAT16, 256, 256));
        assertArrayEquals(Tensor.ones(DType.FLOAT16, 64, 256).toFloatArray(), actual.toFloatArray());
    }

    @Test void broadcastRowsUseTheirLogicalValues() {
        Tensor a = Tensor.backend().expand(values(1, 257), 64, 257);
        compare(a, values(257, 129));
    }

    @Test void compactForwardKeepsFloat32Adjoints() {
        Tensor a = values(64, 256).requiresGrad(true), b = values(256, 512).requiresGrad(true);
        Tensor left = a.to(DType.FLOAT32).detach().requiresGrad(true);
        Tensor right = b.to(DType.FLOAT32).detach().requiresGrad(true);
        a.matmul(b).sumTensor().backward();
        left.matmul(right).sumTensor().backward();
        assertEquals(DType.FLOAT32, a.grad().dtype());
        assertEquals(DType.FLOAT32, b.grad().dtype());
        assertArrayEquals(left.grad().toFloatArray(), a.grad().toFloatArray(), 1e-5f);
        assertArrayEquals(right.grad().toFloatArray(), b.grad().toFloatArray(), 1e-5f);
    }

    private static void compare(Tensor a, Tensor b) {
        Tensor reference = a.to(DType.FLOAT32).matmul(b.to(DType.FLOAT32));
        Tensor result = a.matmul(b);
        assertEquals(DType.FLOAT16, result.dtype());
        float[] expected = reference.toFloatArray(), actual = result.toFloatArray();
        for (int index = 0; index < actual.length; index++) {
            assertEquals(expected[index], actual[index], .002f + Math.abs(expected[index]) * .002f);
        }
    }

    private static Tensor values(int rows, int columns) {
        return Tensor.random(DType.FLOAT16, new Random(481 + rows + columns), rows, columns).multiplyScalar(.25f);
    }
}
