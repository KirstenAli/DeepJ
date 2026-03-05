
package io.github.kirstenali.deepj.activations;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class ActivationsTest {

    @Test
    void relu_forwardBackward() {
        ReLU relu = new ReLU();
        Tensor x = TestSupport.tensor(new double[][]{
                {-1, 0, 2},
                {3, -4, 5}
        });

        Tensor y = relu.forward(x);
        TestSupport.assertTensorAllClose(y, TestSupport.tensor(new double[][]{
                {0, 0, 2},
                {3, 0, 5}
        }), 1e-12);

        Tensor gradOut = TestSupport.tensor(new double[][]{
                {1, 1, 1},
                {2, 2, 2}
        });
        Tensor gx = relu.backward(gradOut);
        TestSupport.assertTensorAllClose(gx, TestSupport.tensor(new double[][]{
                {0, 0, 1},
                {2, 0, 2}
        }), 1e-12);
    }

    @Test
    void sigmoid_outputsInRange_andBackwardNonNegativeForPositiveUpstream() {
        Sigmoid s = new Sigmoid();
        Tensor x = TestSupport.tensor(new double[][]{{-10, 0, 10}});
        Tensor y = s.forward(x);

        for (int c = 0; c < y.cols; c++) {
            Assertions.assertTrue(y.data[0][c] > 0.0 && y.data[0][c] < 1.0);
        }

        Tensor gradOut = TestSupport.tensor(new double[][]{{1, 1, 1}});
        Tensor gx = s.backward(gradOut);
        for (int c = 0; c < gx.cols; c++) {
            Assertions.assertTrue(gx.data[0][c] >= 0.0, "sigmoid' should be >= 0");
        }
    }

    @Test
    void tanh_isOddFunctionApprox() {
        Tanh t = new Tanh();
        Tensor x1 = TestSupport.tensor(new double[][]{{0.5, -0.5}});
        Tensor y1 = t.forward(x1);
        Assertions.assertEquals(y1.data[0][0], -y1.data[0][1], 1e-12);
    }

    @Test
    void gelu_isSmooth_andMonotonicAroundZero() {
        GELU g = new GELU();
        Tensor x = TestSupport.tensor(new double[][]{{-1e-3, 0.0, 1e-3}});
        Tensor y = g.forward(x);

        Assertions.assertTrue(y.data[0][0] < y.data[0][1]);
        Assertions.assertTrue(y.data[0][1] < y.data[0][2]);
    }

    @Test
    void softmax_rowsSumTo1() {
        Softmax sm = new Softmax();
        Tensor logits = TestSupport.tensor(new double[][]{
                {1, 2, 3},
                {-1, 0, 1}
        });
        Tensor p = sm.forward(logits);

        for (int r = 0; r < p.rows; r++) {
            double sum = 0.0;
            for (int c = 0; c < p.cols; c++) sum += p.data[r][c];
            Assertions.assertEquals(1.0, sum, 1e-9);
        }

        // backward should require forward
        Softmax sm2 = new Softmax();
        Assertions.assertThrows(IllegalStateException.class, () -> sm2.backward(p));
    }
}
