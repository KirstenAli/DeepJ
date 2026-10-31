package io.github.kirstenali.deepj.loss;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Arrays;
import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertTensorGradients;
import static org.junit.jupiter.api.Assertions.*;

/** Independent scalar references shared by CPU and actual Metal tests. */
public final class NdMseChecks {
    private NdMseChecks() { }

    public static void ranksAndViews(DType dtype) {
        for (int rank = 0; rank <= 6; rank++) {
            int[] shape = new int[rank];
            Arrays.fill(shape, 2);
            Tensor predicted = Tensor.from(values(1 << rank, 7, 0.25f), dtype, shape);
            Tensor target = Tensor.from(values(1 << rank, 3, 0.5f), dtype, shape);
            reference(predicted, target);
            if (rank > 1) reference(predicted.transpose(0, rank - 1), target.transpose(0, rank - 1));
            if (rank > 0) reference(predicted.slice(0, 1, 2), target.slice(0, 1, 2));
        }
    }

    public static void bothInputGradients(DType dtype) {
        Tensor predicted = Tensor.from(values(8, 7, 0.25f), dtype, 2, 2, 2).requiresGrad(true);
        Tensor target = Tensor.from(values(8, 3, 0.5f), dtype, 2, 2, 2).requiresGrad(true);
        float[] expected = gradient(predicted.toFloatArray(), target.toFloatArray());
        new MSELoss().forward(predicted, target).multiplyScalar(2).backward();
        for (int index = 0; index < expected.length; index++) { expected[index] *= 2; }
        assertArrayEquals(expected, predicted.grad().toFloatArray());
        for (int index = 0; index < expected.length; index++) { expected[index] = 0.0f - expected[index]; }
        assertArrayEquals(expected, target.grad().toFloatArray());
        assertEquals(DType.FLOAT32, predicted.grad().dtype());
        assertEquals(DType.FLOAT32, target.grad().dtype());
    }

    public static void mixedAndMatrixOrder(DType dtype) {
        Tensor predicted = Tensor.from(values(6, 7, 0.25f), dtype, 2, 3);
        Tensor target = Tensor.from(values(6, 3, 0.5f), 2, 3);
        var mse = new MSELoss();
        Tensor difference = predicted.to(DType.FLOAT32).subtract(target);
        assertEquals(difference.multiply(difference).sumTensor().divideScalar(6).item(),
                mse.forward(predicted, target).item());
        assertArrayEquals(new int[]{1, 1}, mse.forward(predicted, target).shape());
        reference(predicted, target);
        reference(target, predicted);
    }

    public static void specialValues() {
        for (DType dtype : DType.values()) {
            Tensor predicted = Tensor.from(new float[]{1024, -1024}, dtype, 1, 2);
            Tensor target = Tensor.from(new float[]{-1024, 1024}, dtype, 1, 2);
            assertEquals(4_194_304f, new MSELoss().loss(predicted, target));
            assertEquals(DType.FLOAT32, new MSELoss().gradient(predicted, target).dtype());
        }
        Tensor nonfinite = Tensor.from(new float[]{Float.NaN, Float.POSITIVE_INFINITY});
        assertTrue(Float.isNaN(new MSELoss().loss(nonfinite, nonfinite)));
    }

    public static void invalidInputs() {
        Tensor value = new Tensor(2, 2);
        Tensor[] invalid = {null, new Tensor(0, 2), new Tensor(1, 2), new Tensor(4)};
        for (Tensor other : invalid) {
            var mse = new MSELoss();
            assertThrows(IllegalArgumentException.class, () -> mse.forward(value, other));
            assertThrows(IllegalArgumentException.class, () -> mse.loss(value, other));
            assertThrows(IllegalArgumentException.class, () -> mse.gradient(value, other));
        }
        assertThrows(IllegalArgumentException.class, () -> new MSELoss().forward(null, value));
        assertThrows(IllegalArgumentException.class, () -> new MSELoss().gradient(new Tensor(0), new Tensor(0)));
        assertArrayEquals(new float[4], value.toFloatArray());
    }

    public static void finiteDifferences() {
        Tensor predicted = Tensor.from(new float[]{0.2f, -0.4f, 0.7f, 1.1f}, 1, 2, 2);
        Tensor target = Tensor.from(new float[]{0.1f, 0.3f, -0.1f, 0.2f}, 1, 2, 2);
        var mse = new MSELoss();
        Tensor upstream = Tensor.from(new float[]{0.7f}, new int[0]);
        assertTensorGradients(predicted, value -> mse.forward(value, target), upstream, 1e-3f, 1e-4f, 1e-3f);
        assertTensorGradients(target, value -> mse.forward(predicted, value), upstream, 1e-3f, 1e-4f, 1e-3f);
    }

    private static void reference(Tensor predicted, Tensor target) {
        float[] before = predicted.toFloatArray(), actual = target.toFloatArray();
        double sum = 0;
        for (int index = 0; index < before.length; index++) {
            double difference = (double) before[index] - actual[index];
            sum += difference * difference;
        }
        var mse = new MSELoss();
        assertEquals((float) (sum / before.length), mse.loss(predicted, target), 1e-7f);
        checkOutputs(predicted, target, gradient(before, actual));
        assertArrayEquals(before, predicted.toFloatArray());
        assertArrayEquals(actual, target.toFloatArray());
    }

    private static void checkOutputs(Tensor predicted, Tensor target, float[] expectedGradient) {
        var mse = new MSELoss();
        Tensor result = mse.forward(predicted, target), gradient = mse.gradient(predicted, target);
        assertEquals(DType.FLOAT32, result.dtype());
        assertEquals(DType.FLOAT32, gradient.dtype());
        assertArrayEquals(predicted.shape(), gradient.shape());
        assertArrayEquals(expectedGradient, gradient.toFloatArray(), 1e-7f);
    }

    private static float[] gradient(float[] predicted, float[] target) {
        float[] result = new float[predicted.length];
        for (int index = 0; index < result.length; index++) {
            result[index] = (predicted[index] - target[index]) * (2.0f / result.length);
        }
        return result;
    }

    private static float[] values(int count, int period, float scale) {
        float[] result = new float[count];
        for (int index = 0; index < count; index++) { result[index] = (index % period - period / 2) * scale; }
        return result;
    }
}
