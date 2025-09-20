package org.DeepJ;

import org.DeepJ.ann.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class SoftmaxTest {

    @Test
    public void testSoftmaxForward() {
        Tensor input = new Tensor(new double[][] {
                {1.0, 2.0, 3.0}
        });

        Tensor softmax = Tensor.softmaxRows(input);
        double[] expected = {0.0900, 0.2447, 0.6652};
        double[] actual = softmax.data[0];

        for (int i = 0; i < expected.length; i++) {
            double diff = Math.abs(expected[i] - actual[i]);
            assertTrue(diff < 1e-3, String.format(
                    "Softmax forward failed at index %d: expected %.4f, got %.4f",
                    i, expected[i], actual[i]));
        }
    }

    @Test
    public void testSoftmaxBackwardGradientOfSum() {
        Tensor input = new Tensor(new double[][] {
                {1.0, 2.0, 3.0}
        });

        Tensor softmax = Tensor.softmaxRows(input);
        Tensor dL_dSoftmax = new Tensor(new double[][] {
                {1.0, 1.0, 1.0}
        });

        Tensor gradInput = Tensor.softmaxBackward(dL_dSoftmax, softmax);
        double[] grad = gradInput.data[0];

        for (int j = 0; j < grad.length; j++) {
            assertEquals(0.0, grad[j], 1e-6,
                    String.format("Softmax backward failed at index %d: expected ≈ 0, got %.6f", j, grad[j]));
        }
    }
}
