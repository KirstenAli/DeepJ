package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class Qwen3ModelTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }
    static Qwen3Config tiny() { return new Qwen3Config(19, 16, 12, 4, 2, 2, 20, 4, 1_000_000, 0.02f, 1); }

    @Test void officialPresetHasTheExpectedGeometryAndTiedParameterCountWithoutAllocating() {
        Qwen3Config config = Qwen3Config.onePointSevenB();
        assertEquals(1_720_574_976L, config.parameterCount());
        assertEquals(128, config.headDim());
        assertEquals(28, config.nLayers());
        assertEquals(1_000_000, config.ropeTheta());
    }

    @ParameterizedTest @EnumSource(DType.class)
    void tensorArithmeticMatchesAnIndependentDoublePrecisionDecoder(DType dtype) {
        Qwen3 model = new Qwen3(tiny(), 17, dtype);
        int[] tokens = {1, 5, 7, 3};
        float[] expected = new Qwen3Reference(model).logits(tokens, tokens.length);
        assertArrayEquals(expected, model.forward(tokens).toFloatArray(), 0.00002f);
        assertEquals(24, model.parameters().size());
        assertEquals(model.config().parameterCount(),
                model.parameters().stream().mapToLong(p -> p.value.numel()).sum());
    }

    @Test void batchedOutputsEqualIndependentSequencesAndFutureTokensCannotChangeEarlierLogits() {
        Qwen3 model = new Qwen3(tiny(), 20);
        int[][] tokens = {{1, 2, 3}, {4, 5, 6}};
        Tensor batch = model.forwardBatch(tokens);
        for (int row = 0; row < tokens.length; row++) {
            assertArrayEquals(model.forward(tokens[row]).toFloatArray(), batch.select(0, row).toFloatArray(), 1e-6f);
        }
        assertArrayEquals(model.forward(new int[]{1, 2, 9}).slice(0, 0, 2).toFloatArray(),
                model.forward(tokens[0]).slice(0, 0, 2).toFloatArray());
    }

    @Test void everyParameterGradientIncludingTiedEmbeddingContributionsMatchesFiniteDifferences() {
        Qwen3 model = new Qwen3(tiny(), 4);
        Tensor upstream = Tensor.random(3, 19, new Random(8));
        model.forward(new int[]{1, 2, 3}).backward(upstream);
        for (var parameter : model.parameters()) {
            float[] values = parameter.value.toFloatArray(), gradients = parameter.grad().toFloatArray();
            int index = largest(gradients);
            float plus = perturb(model, parameter.value, index, values[index] + 0.001f, upstream);
            float minus = perturb(model, parameter.value, index, values[index] - 0.001f, upstream);
            set(parameter.value, index, values[index]);
            assertEquals((plus - minus) / 0.002f, gradients[index],
                    0.0001f + Math.abs(gradients[index]) * 0.01f, parameter.name());
        }
    }

    private static int largest(float[] values) {
        int result = 0;
        for (int index = 1; index < values.length; index++) {
            if (Math.abs(values[index]) > Math.abs(values[result])) result = index;
        }
        return result;
    }

    private static float perturb(Qwen3 model, Tensor parameter, int index, float value, Tensor upstream) {
        set(parameter, index, value);
        try (var ignored = GradMode.noGrad()) {
            return model.forward(new int[]{1, 2, 3}).multiply(upstream).sumTensor().item();
        }
    }

    private static void set(Tensor parameter, int index, float value) {
        try (var ignored = GradMode.noGrad()) {
            parameter.set(index / parameter.matrixCols(), index % parameter.matrixCols(), value);
        }
    }

    @ParameterizedTest @EnumSource(DType.class)
    void existingTrainingAndAdamWImproveTheLossWithFloat32Gradients(DType dtype) {
        Qwen3 model = new Qwen3(tiny(), 3, dtype);
        Batch batch = new Batch(new int[][]{{1, 2, 3}, {4, 5, 6}}, new int[][]{{2, 3, 4}, {5, 6, 7}});
        var optimizer = AdamW.defaultAdamW(0.005f);
        var trainer = CausalLMTraining.trainer(model, size -> batch, optimizer);
        float first = trainer.trainStep(2), last = first;
        for (int step = 0; step < 9; step++) { last = trainer.trainStep(2); }
        assertTrue(last < first, first + " -> " + last);
        assertEquals(10, optimizer.stepCount());
        for (var parameter : model.parameters()) {
            assertEquals(DType.FLOAT32, parameter.grad().dtype());
            assertTrue(Arrays.stream(ReferenceMath.doubles(parameter.grad().toFloatArray()))
                    .allMatch(Double::isFinite));
        }
    }
}
