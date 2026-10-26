package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

/** Queue changing stack arguments before flushing to catch argument lifetime and binding mistakes. */
class MetalArgumentBindingTest extends MetalDifferentialTestSupport {

    @Test
    void queuedScalarsAndClampRangesKeepTheirOwnValues() {
        Tensor input = random(3, 33, 2026);
        List<Result> results = new ArrayList<>();
        for (int index = 0; index < 128; index++) {
            float scalar = (index - 64) * 0.125f;
            results.add(new Result(cpu.addScalar(input, scalar), metal.addScalar(input, scalar)));
            results.add(new Result(cpu.multiplyScalar(input, scalar), metal.multiplyScalar(input, scalar)));
            results.add(new Result(cpu.clamp(input, scalar, scalar + 0.5f),
                    metal.clamp(input, scalar, scalar + 0.5f)));
        }
        results.forEach(result -> result.verify(0));
    }

    @Test
    void queuedDimensionsSurviveChangesBetweenDispatches() {
        List<Result> results = new ArrayList<>();
        for (int index = 0; index < 64; index++) {
            Tensor input = random(1 + index % 7, 3 + index % 33, index);
            Tensor row = random(1, input.matrixCols(), index + 100);
            results.add(new Result(cpu.transpose(input), metal.transpose(input)));
            results.add(new Result(cpu.addRowVector(input, row), metal.addRowVector(input, row)));
            results.add(new Result(cpu.sumRows(input), metal.sumRows(input)));
            results.add(new Result(cpu.softmaxRows(input), metal.softmaxRows(input)));
        }
        results.forEach(result -> result.verify(1e-5f));
    }

    @Test
    void queuedNormalizationAndAttentionKeepTheirOwnStructures() {
        List<Result> results = new ArrayList<>();
        for (int index = 0; index < 64; index++) {
            Tensor input = random(3 + index % 7, 3 + index % 7, index);
            Tensor gamma = positive(1, input.matrixCols(), index + 100);
            float epsilon = 0.0001f + index * 0.001f;
            float scale = 0.125f + index * 0.01f;
            results.add(new Result(cpu.rmsNorm(input, gamma, epsilon).output(),
                    metal.rmsNorm(input, gamma, epsilon).output()));
            results.add(new Result(cpu.causalSoftmax(input, input.matrixRows(), scale),
                    metal.causalSoftmax(input, input.matrixRows(), scale)));
        }
        results.forEach(result -> result.verify(1e-5f));
    }

    @Test
    void queuedAdamUpdatesKeepAllSevenArguments() {
        List<Result> results = new ArrayList<>();
        for (int index = 0; index < 64; index++) {
            Tensor weights = random(3, 33, index);
            Tensor gradient = random(3, 33, index + 100);
            var expected = adam(cpu, new Tensor(weights), gradient, index);
            var actual = adam(metal, new Tensor(weights), gradient, index);
            for (int state = 0; state < expected.size(); state++) {
                results.add(new Result(expected.get(state), actual.get(state)));
            }
        }
        results.forEach(result -> result.verify(1e-5f));
    }

    private List<Tensor> adam(TensorBackend backend, Tensor weights, Tensor gradient, int index) {
        Tensor first = random(weights.matrixRows(), weights.matrixCols(), index + 200);
        Tensor second = positive(weights.matrixRows(), weights.matrixCols(), index + 300);
        backend.adamWUpdate(weights, gradient, first, second, 0.0001f * (index + 1),
                0.8f + index * 0.001f, 0.95f + index * 0.0001f, 1e-8f + index * 1e-5f,
                index * 0.001f, 0.1f + index * 0.01f, 0.001f + index * 0.001f);
        return List.of(weights, first, second);
    }

    private record Result(Tensor expected, Tensor actual) {
        void verify(float tolerance) {
            assertTensorClose(expected, actual, tolerance, tolerance);
        }
    }
}
