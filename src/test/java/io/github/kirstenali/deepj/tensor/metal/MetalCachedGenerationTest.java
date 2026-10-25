package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.testing.InferenceTestModels;
import io.github.kirstenali.deepj.testing.InferenceTestModels.Kind;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class MetalCachedGenerationTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @MethodSource("modelsAndCapacities")
    void incrementalGpuLogitsMatchFullRecomputationAndSurviveTemporaryCleanup(Kind kind, int capacity) {
        DecoderOnlyModel model = InferenceTestModels.create(kind, capacity, 128);
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

    private static Stream<Arguments> modelsAndCapacities() {
        return Arrays.stream(Kind.values()).flatMap(kind -> Stream.of(288, 1024)
                .map(capacity -> Arguments.of(kind, capacity)));
    }

    @ParameterizedTest
    @EnumSource(Kind.class)
    void repeatedCachedRequestsHaveIdenticalGpuLogits(Kind kind) {
        DecoderOnlyModel model = InferenceTestModels.create(kind, 1024, 128);
        int[] tokens = randomTargets(70, 31, 503);
        List<float[]> expected = decoded(model, tokens), actual = decoded(model, tokens);
        for (int index = 0; index < expected.size(); index++) {
            assertArrayEquals(expected.get(index), actual.get(index));
        }
    }

    private List<float[]> decoded(DecoderOnlyModel model, int[] tokens) {
        List<float[]> result = new ArrayList<>();
        try (var session = model.openGeneration()) {
            for (int length = 64; length <= tokens.length; length++) {
                result.add(session.apply(Arrays.copyOf(tokens, length)).rowData(0));
                metal.releaseTemporaryResources();
            }
        }
        return result;
    }

    private static Tensor last(Tensor logits) {
        return Tensor.from2D(new float[][]{logits.rowData(logits.rows - 1)});
    }

    @ParameterizedTest
    @EnumSource(Kind.class)
    void slidingContextAndInterleavedRequestsMatchFullGpuRecomputation(Kind kind) {
        DecoderOnlyModel model = InferenceTestModels.create(kind, 288, 128);
        int[] tokens = randomTargets(294, 31, 509);
        try (var first = model.openGeneration(); var second = model.openGeneration();
             var ignored = GradMode.noGrad()) {
            first.apply(Arrays.copyOf(tokens, 286)).materialize();
            second.apply(new int[]{3, 7}).materialize();
            for (int length = 287; length <= tokens.length; length++) {
                int[] window = Arrays.copyOfRange(tokens, Math.max(0, length - 288), length);
                assertTensorClose(last(model.forward(window)), last(first.apply(window)), 2e-5f, 2e-4f);
                metal.releaseTemporaryResources();
            }
            int[] other = {3, 7, 11};
            assertTensorClose(last(model.forward(other)), last(second.apply(other)), 2e-5f, 2e-4f);
        }
    }
}
