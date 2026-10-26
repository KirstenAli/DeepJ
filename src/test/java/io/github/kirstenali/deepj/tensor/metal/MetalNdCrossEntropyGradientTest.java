package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import java.util.List;
import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertTensorGradients;
import static org.junit.jupiter.api.Assertions.*;

class MetalNdCrossEntropyGradientTest extends MetalDifferentialTestSupport {
    @Test
    void maskedNdAndStridedClassAxisMatchActualGpuFiniteDifferences() {
        Tensor logits = Tensor.from(new float[]{0.2f, -0.4f, 0.7f, 1.1f, 0.3f, -0.2f}, 1, 2, 3);
        assertTensorGradients(logits, input -> CrossEntropyLoss.forward(input, new int[]{2, 0},
                new boolean[]{false, true}), Tensor.from(new float[]{0.7f}, new int[0]), 1e-3f, 1e-4f, 1e-3f);
        assertTensorGradients(logits, input -> CrossEntropyLoss.forward(input.permute(0, 2, 1),
                new int[]{0, 1, 0}, null), Tensor.from(new float[]{1}, new int[0]), 1e-3f, 1e-4f, 1e-3f);
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void typedLeavesGetUnroundedFloat32GradientsAndGpuScalarUpstreamWorks(DType dtype) {
        Tensor logits = Tensor.from(new float[]{0.2f, -0.4f, 0.7f, 1.1f, 0.3f, -0.2f}, dtype, 1, 2, 3)
                .requiresGrad(true);
        Tensor expected = metal.multiplyScalar(metal.crossEntropyGradient(logits, new int[]{2, 0}), 0.7f);
        Tensor upstream = metal.copy(Tensor.from(new float[]{0.7f}, new int[0]));
        CrossEntropyLoss.forward(logits, new int[]{2, 0}, null).backward(upstream);
        assertEquals(DType.FLOAT32, logits.grad().dtype());
        assertArrayEquals(expected.toFloatArray(), logits.grad().toFloatArray());
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void optionalCompactCrossEntropyTrainingImprovesAndRestoredMastersContinueExactly(DType dtype) {
        Parameter logits = new Parameter(Tensor.from(new float[]{0.1f, -0.1f, 0.2f, 0.3f, -0.4f, 0.2f}, dtype,
                1, 2, 3));
        AdamW optimizer = AdamW.defaultAdamW(0.01f);
        float initial = stepLoss(logits).item();
        for (int step = 0; step < 25; step++) { train(logits, optimizer); }
        assertTrue(stepLoss(logits).item() < initial);
        Parameter restored = new Parameter(logits.value.detach());
        AdamW continuation = AdamW.defaultAdamW(0.01f);
        continuation.restoreState(List.of(restored), optimizer.state(List.of(logits)));
        train(logits, optimizer);
        train(restored, continuation);
        assertArrayEquals(logits.value.toFloatArray(), restored.value.toFloatArray());
        assertArrayEquals(optimizer.state(List.of(logits)).masterWeights().get(0).toFloatArray(),
                continuation.state(List.of(restored)).masterWeights().get(0).toFloatArray());
    }

    private static Tensor stepLoss(Parameter logits) {
        return CrossEntropyLoss.forward(logits.value, new int[]{2, 1}, new boolean[]{true, true});
    }
    private static void train(Parameter logits, AdamW optimizer) {
        logits.zeroGrad();
        stepLoss(logits).backward();
        optimizer.step(List.of(logits));
    }
}
