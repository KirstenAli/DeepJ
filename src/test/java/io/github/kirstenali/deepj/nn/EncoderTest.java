package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import java.util.Random;
import static io.github.kirstenali.deepj.nn.NeuralTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

class EncoderTest extends NeuralTest {
    @Test void attentionMatchesIndependentSoftmaxWeightedSumsWithPadding() {
        EncoderAttention attention = new EncoderAttention(2, 1, 0, new Random(1), ParameterFactory.float32());
        identity(attention);
        Tensor input = Tensor.from(new float[]{1, 0, 0, 2, 99, 99}, 1, 3, 2);
        boolean[][] valid = {{true, true, false}};
        Tensor output = attention.forward(input, valid);
        for (int time = 0; time < 2; time++) {
            double a = Math.exp(input.get(0, time, 0) / Math.sqrt(2));
            double b = Math.exp(2 * input.get(0, time, 1) / Math.sqrt(2));
            assertEquals(a / (a + b), output.get(0, time, 0), 1e-6);
            assertEquals(2 * b / (a + b), output.get(0, time, 1), 1e-6);
        }
        assertArrayEquals(new float[]{0, 0}, output.select(0, 0).select(0, 2).toFloatArray());
        gradients(attention, values(1, 2, 2));
    }

    @Test void encoderGradientsIncludeEveryBlockAndExistingLayerNormParameters() {
        TransformerEncoder encoder = encoder(1, DType.FLOAT32);
        initialize(encoder);
        encoder.eval();
        gradients(encoder, values(1, 2, 2));
        TransformerEncoderLayer block = new TransformerEncoderLayer(config(1), new Random(1),
                ParameterFactory.float32());
        gradients(block, values(1, 2, 2));
        assertEquals(config(1), encoder.config());
        assertArrayEquals(new int[]{1, 2, 2}, block.forward(values(1, 2, 2), new boolean[][]{{true, false}}).shape());
    }

    @Test void masksExcludePaddedKeysAndQueriesAndMeanPoolingUsesValidCounts() {
        TransformerEncoder encoder = encoder(2, DType.FLOAT32);
        encoder.eval();
        Tensor input = values(2, 3, 2).requiresGrad(true);
        boolean[][] valid = {{true, false, true}, {true, true, false}};
        Tensor output = encoder.forward(input, valid);
        Tensor pooled = encoder.meanPool(input, valid);
        assertArrayEquals(output.sum(1, false).divideScalar(2).toFloatArray(), pooled.toFloatArray(), 1e-6f);
        output.multiply(values(output.shape())).sumTensor().backward();
        assertArrayEquals(new float[]{0, 0}, input.grad().select(0, 0).select(0, 1).toFloatArray());
        assertArrayEquals(new float[]{0, 0}, output.select(0, 1).select(0, 2).toFloatArray());
        Tensor changed = input.detach();
        set(changed, 2, 500);
        set(changed, 3, -900);
        assertArrayEquals(output.toFloatArray(), encoder.forward(changed, valid).toFloatArray(), 1e-6f);
    }

    @Test void batchesAreIndependentAndCompactParametersUseTheSameEncoder() {
        for (DType dtype : DType.values()) {
            TransformerEncoder encoder = encoder(1, dtype);
            encoder.eval();
            Tensor input = values(2, 2, 2).to(dtype), together = encoder.forward(input);
            for (int sample = 0; sample < 2; sample++) {
                Tensor alone = encoder.forward(input.slice(0, sample, sample + 1));
                assertArrayEquals(together.select(0, sample).toFloatArray(), alone.toFloatArray(), 1e-6f);
            }
            assertArrayEquals(new int[]{2, 2}, encoder.meanPool(input, null).shape());
        }
    }

    @Test void trainingModePropagatesToAttentionAndResidualDropout() {
        TransformerEncoderConfig config = new TransformerEncoderConfig(4, 2, 6, 2, 0.5f);
        TransformerEncoder encoder = new TransformerEncoder(config, new Random(1));
        Tensor input = values(2, 3, 4);
        encoder.eval();
        assertArrayEquals(encoder.forward(input).toFloatArray(), encoder.forward(input).toFloatArray());
        encoder.train();
        assertFalse(java.util.Arrays.equals(encoder.forward(input).toFloatArray(),
                encoder.forward(input).toFloatArray()));
    }

    private static TransformerEncoderConfig config(int layers) {
        return new TransformerEncoderConfig(2, 1, 3, layers, 0);
    }

    private static TransformerEncoder encoder(int layers, DType dtype) {
        return new TransformerEncoder(config(layers), new Random(12), new ParameterFactory(dtype));
    }

    private static void identity(EncoderAttention attention) {
        try (var ignored = GradMode.noGrad()) {
            for (var parameter : attention.parameters()) {
                Tensor value = parameter.value;
                value.copyFrom(value.size(0) == 2 ? Tensor.from(new float[]{1, 0, 0, 1}, 2, 2) : new Tensor(1, 2));
            }
        }
    }
}
