package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class ModelNormTest {
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
    void centeredInferenceMatchesTensorCompositionForWideRowsAndOffsetStridedViews(DType dtype) {
        float[] data = new float[3 * 2 * 257];
        for (int index = 0; index < data.length; index++) { data[index] = (float) Math.sin(index * 0.31); }
        Tensor input = Tensor.from(data, dtype, 3, 2, 257).slice(0, 1, 3);
        try (var ignored = GradMode.noGrad()) {
            for (Tensor source : new Tensor[]{input, input.permute(0, 2, 1)}) {
                float[] expected = ModelOps.norm(source, 1e-5f).toFloatArray();
                Tensor output = ModelInference.batch(() -> ModelOps.norm(source, 1e-5f));
                assertEquals(dtype, output.dtype());
                float tolerance = dtype == DType.FLOAT32 ? 3e-6f : dtype == DType.BFLOAT16 ? 0.02f : 0.002f;
                assertArrayEquals(expected, output.toFloatArray(), tolerance);
            }
        }
    }
}
