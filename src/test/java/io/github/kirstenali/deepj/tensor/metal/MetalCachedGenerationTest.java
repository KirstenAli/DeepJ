package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.testing.InferenceTestModels;
import io.github.kirstenali.deepj.testing.InferenceTestModels.Kind;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Arrays;

class MetalCachedGenerationTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(Kind.class)
    void incrementalGpuLogitsMatchFullRecomputationAndSurviveTemporaryCleanup(Kind kind) {
        DecoderOnlyModel model = InferenceTestModels.create(kind, 288, 128);
        int[] tokens = randomTargets(264, 31, 449);
        try (var session = model.openGeneration(); var ignored = GradMode.noGrad()) {
            session.apply(Arrays.copyOf(tokens, 256)).materialize();
            metal.releaseTemporaryResources();
            for (int length = 257; length <= tokens.length; length++) {
                int[] prefix = Arrays.copyOf(tokens, length);
                Tensor expected = last(model.forward(prefix));
                Tensor actual = last(session.apply(prefix));
                assertTensorClose(expected, actual, 2e-5f, 2e-4f);
                metal.releaseTemporaryResources();
            }
        }
    }

    private static Tensor last(Tensor logits) {
        return Tensor.from2D(new float[][]{logits.rowData(logits.rows - 1)});
    }
}
