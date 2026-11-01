package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import java.util.Arrays;
import static io.github.kirstenali.deepj.nn.NeuralTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

class NormalizationTest extends NeuralTest {
    @Test void batchNormalizationUsesPopulationVarianceAndUnbiasedRunningVariance() {
        BatchNorm norm = new BatchNorm(2, 1e-5f, 0.5f, ParameterFactory.float32());
        Tensor input = Tensor.from(new float[]{1, 3, 10, 14, 5, 7, 18, 22}, 2, 2, 2);
        Tensor output = norm.forward(input);
        assertEquals(-3 / Math.sqrt(5.00001), output.get(0, 0, 0), 1e-6);
        assertEquals(-6 / Math.sqrt(20.00001), output.get(0, 1, 0), 1e-6);
        assertArrayEquals(new float[]{2, 8}, norm.runningMean().toFloatArray());
        assertArrayEquals(new float[]{0.5f + 10.0f / 3, 0.5f + 40.0f / 3},
                norm.runningVariance().toFloatArray(), 1e-6f);
        assertEquals(2, norm.buffers().size());
        norm.eval();
        assertEquals((1 - 2) / Math.sqrt(0.5 + 10.0 / 3 + 1e-5), norm.forward(input).get(0, 0, 0), 1e-6);
        assertArrayEquals(new float[]{2, 8}, norm.runningMean().toFloatArray());
    }

    @Test void trainingAndEvaluationGradientsMatchFiniteDifferences() {
        BatchNorm norm = new BatchNorm(2);
        gradients(norm, values(2, 2, 2));
        norm.eval();
        gradients(norm, values(2, 2, 2));
        assertArrayEquals(new int[]{1, 2}, norm.forward(values(1, 2)).shape());
    }

    @Test void dropoutIsSeededScaledDifferentiableAndDisabledByEvaluationMode() {
        Dropout first = new Dropout(0.5f, 17), second = new Dropout(0.5f, 17);
        Tensor input = Tensor.ones(4, 8).requiresGrad(true);
        Tensor output = first.forward(input);
        assertArrayEquals(output.toFloatArray(), second.forward(input).toFloatArray());
        output.sumTensor().backward();
        assertArrayEquals(output.toFloatArray(), input.grad().toFloatArray());
        assertTrue(Arrays.stream(toDoubles(output.toFloatArray())).anyMatch(value -> value == 0));
        assertTrue(Arrays.stream(toDoubles(output.toFloatArray())).anyMatch(value -> value == 2));
        first.eval();
        assertSame(input, first.forward(input));
        first.train();
        try (var ignored = GradMode.noGrad()) { assertNotSame(input, first.forward(input)); }
    }

    @Test void invalidNormalizationParametersAndInputShapesAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> new BatchNorm(0));
        assertThrows(IllegalArgumentException.class,
                () -> new BatchNorm(1, Float.NaN, 0.1f, ParameterFactory.float32()));
        assertThrows(IllegalArgumentException.class, () -> new BatchNorm(1, 0, 0.1f, ParameterFactory.float32()));
        assertThrows(IllegalArgumentException.class, () -> new BatchNorm(1, 1e-5f, -1, ParameterFactory.float32()));
        assertThrows(IllegalArgumentException.class, () -> new BatchNorm(1, 1e-5f, 2, ParameterFactory.float32()));
        assertThrows(IllegalArgumentException.class,
                () -> new BatchNorm(1, 1e-5f, Float.NaN, ParameterFactory.float32()));
        BatchNorm norm = new BatchNorm(2);
        assertThrows(IllegalArgumentException.class, () -> norm.forward(values(1, 2)));
        assertThrows(IllegalArgumentException.class, () -> norm.forward(values(2, 1)));
        assertThrows(IllegalArgumentException.class, () -> norm.forward(values(2)));
        assertThrows(IllegalArgumentException.class, () -> norm.forward(null));
        assertThrows(IllegalArgumentException.class, () -> norm.forward(new Tensor(0, 2)));
    }

    private static double[] toDoubles(float[] values) {
        double[] result = new double[values.length];
        for (int index = 0; index < values.length; index++) { result[index] = values[index]; }
        return result;
    }
}
