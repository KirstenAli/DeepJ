package io.github.kirstenali.deepj.nn.transformer.gemma2;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.nn.ModelInference;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class Gemma2SchedulingTest {
    @BeforeEach void backend() {
        Tensor.setBackend(Boolean.getBoolean("gemma.test.metal") ? new MetalBackend() : new CpuBackend());
    }

    @ParameterizedTest @EnumSource(DType.class)
    void batchedAndStagedInferencePreserveEveryLogitAndReusableWeight(DType dtype) throws Exception {
        try (var model = Gemma2Pretrained.load(Gemma2ReferenceTest.ROOT, dtype);
             var ignored = GradMode.noGrad()) {
            int[] tokens = {2, 3, 4, 5};
            float[] expected = model.forward(tokens).toFloatArray();
            for (int iteration = 0; iteration < 4; iteration++) {
                Tensor result = ModelInference.batch(() -> model.forward(tokens));
                assertArrayEquals(expected, result.toFloatArray());
                assertFalse(result.requiresGrad());
                Tensor.backend().releaseTemporaryResources();
            }
            assertArrayEquals(expected, model.forward(tokens).toFloatArray());
            assertTrue(model.parameters().stream().allMatch(parameter -> parameter.value.retainsDeviceBuffer()));
        }
    }
}
