package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.util.Random;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class ModelLinearAttentionTest {
    private final TensorBackend previous = Tensor.backend();

    @BeforeEach void backend() {
        boolean metal = Boolean.getBoolean("sana.test.metal") || Boolean.getBoolean("deepj.requireMetal");
        Tensor.setBackend(metal ? new MetalBackend() : new CpuBackend());
    }

    @AfterEach void restore() {
        Tensor.backend().releaseResources();
        Tensor.setBackend(previous);
    }

    @ParameterizedTest @EnumSource(DType.class)
    void combinesValuesAndDenominatorWithIndependentBatchHeadAndSequenceDimensions(DType dtype) {
        Tensor query = values(dtype, 2, 3, 7, 4), key = values(dtype, 2, 3, 5, 4);
        Tensor value = values(dtype, 2, 3, 5, 6);
        float[] expected = ModelOps.linearAttention(query, key, value, 1e-15f).toFloatArray();
        try (var ignored = GradMode.noGrad()) {
            Tensor output = ModelOps.linearAttention(query, key, value, 1e-15f);
            assertArrayEquals(new int[]{2, 3, 7, 6}, output.shape());
            float tolerance = dtype == DType.FLOAT32 ? 3e-6f : dtype == DType.BFLOAT16 ? 0.01f : 1e-3f;
            assertArrayEquals(expected, output.toFloatArray(), tolerance);
            Tensor zero = ModelOps.linearAttention(query.multiplyScalar(0), key, value, 1e-15f);
            assertArrayEquals(new float[zero.numel()], zero.toFloatArray(), 0);
        }
    }

    private static Tensor values(DType dtype, int... shape) {
        int count = java.util.Arrays.stream(shape).reduce(1, Math::multiplyExact);
        float[] values = new float[count];
        Random random = new Random(count);
        for (int index = 0; index < count; index++) { values[index] = random.nextFloat() - 0.5f; }
        return Tensor.from(values, dtype, shape);
    }
}
