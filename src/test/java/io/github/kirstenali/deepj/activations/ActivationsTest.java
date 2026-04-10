package io.github.kirstenali.deepj.activations;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

public class ActivationsTest {

    @Test
    void relu_forwardBackward() {
        ReLU relu = new ReLU();
        Tensor x = new Tensor(new double[][]{
                {-1, 0, 2},
                {3, -4, 5}
        });

        Tensor y = relu.forward(x);
        TestSupport.assertTensorAllClose(y, new Tensor(new double[][]{
                {0, 0, 2},
                {3, 0, 5}
        }), 1e-12);

        Tensor gradOut = new Tensor(new double[][]{
                {1, 1, 1},
                {2, 2, 2}
        });
        Tensor gx = relu.backward(gradOut);
        TestSupport.assertTensorAllClose(gx, new Tensor(new double[][]{
                {0, 0, 1},
                {2, 0, 2}
        }), 1e-12);
    }

    @Test
    void sigmoid_outputsInRange_andBackwardNonNegativeForPositiveUpstream() {
        Sigmoid s = new Sigmoid();
        Tensor x = new Tensor(new double[][]{{-10, 0, 10}});
        Tensor y = s.forward(x);

        for (int c = 0; c < y.cols; c++) {
            Assertions.assertTrue(y.data[c] > 0.0 && y.data[c] < 1.0);
        }

        Tensor gradOut = new Tensor(new double[][]{{1, 1, 1}});
        Tensor gx = s.backward(gradOut);
        for (int c = 0; c < gx.cols; c++) {
            Assertions.assertTrue(gx.data[c] >= 0.0, "sigmoid' should be >= 0");
        }
    }

    @Test
    void tanh_isOddFunctionApprox() {
        Tanh t = new Tanh();
        Tensor x1 = new Tensor(new double[][]{{0.5, -0.5}});
        Tensor y1 = t.forward(x1);
        Assertions.assertEquals(y1.data[0], -y1.data[1], 1e-12);
    }

    @Test
    void gelu_isSmooth_andMonotonicAroundZero() {
        GELU g = new GELU();
        Tensor x = new Tensor(new double[][]{{-1e-3, 0.0, 1e-3}});
        Tensor y = g.forward(x);

        Assertions.assertTrue(y.data[0] < y.data[1]);
        Assertions.assertTrue(y.data[1] < y.data[2]);
    }

    @Test
    void softmax_rowsSumTo1() {
        Softmax sm = new Softmax();
        Tensor logits = new Tensor(new double[][]{
                {1, 2, 3},
                {-1, 0, 1}
        });
        Tensor p = sm.forward(logits);

        for (int r = 0; r < p.rows; r++) {
            double sum = 0.0;
            for (int c = 0; c < p.cols; c++) sum += p.data[r * p.cols + c];
            Assertions.assertEquals(1.0, sum, 1e-9);
        }

        // backward should require forward
        Softmax sm2 = new Softmax();
        Assertions.assertThrows(IllegalStateException.class, () -> sm2.backward(p));
    }

    @Test
    void sigmoid_backward_matchesFiniteDifference() {
        Tensor x = new Tensor(new double[][]{{-1.2, 0.0, 2.3}});

        // analytic: d/dx sum(sigmoid(x)) = sigmoid'(x)
        Sigmoid s = new Sigmoid();
        Tensor y = s.forward(x);
        Tensor analytic = s.backward(Tensor.ones(y.rows, y.cols));

        // numeric
        Tensor numeric = finiteDiffGradSum(t -> {
            Sigmoid ss = new Sigmoid();
            return ss.forward(t);
        }, x, 1e-6);

        TestSupport.assertTensorAllClose(analytic, numeric, 1e-5);
    }

    @Test
    void tanh_backward_matchesFiniteDifference() {
        Tensor x = new Tensor(new double[][]{{-0.7, 0.2, 1.1}});

        Tanh t = new Tanh();
        Tensor y = t.forward(x);
        Tensor analytic = t.backward(Tensor.ones(y.rows, y.cols));

        Tensor numeric = finiteDiffGradSum(u -> {
            Tanh tt = new Tanh();
            return tt.forward(u);
        }, x, 1e-6);

        TestSupport.assertTensorAllClose(analytic, numeric, 1e-5);
    }

    @Test
    void gelu_backward_matchesFiniteDifference() {
        Tensor x = new Tensor(new double[][]{{-1.5, -0.2, 0.0, 0.4, 2.0}});

        GELU g = new GELU();
        Tensor y = g.forward(x);
        Tensor analytic = g.backward(Tensor.ones(y.rows, y.cols));

        Tensor numeric = finiteDiffGradSum(u -> {
            GELU gg = new GELU();
            return gg.forward(u);
        }, x, 1e-6);

        // GELU is approximate + exp/tanh internally -> slightly looser tolerance
        TestSupport.assertTensorAllClose(analytic, numeric, 1e-4);
    }

    @Test
    void softmax_backward_matchesFiniteDifference_forDotObjective() {
        Tensor logits = new Tensor(new double[][]{
                { 0.3, -1.2, 2.0 },
                { 1.0,  0.5, -0.7 }
        });

        Tensor upstream = new Tensor(new double[][]{
                { 0.7, -0.2, 1.3 },
                { -1.1, 0.4, 0.9 }
        });

        // analytic: grad = softmax.backward(upstream)
        Softmax sm = new Softmax();
        sm.forward(logits);
        Tensor analytic = sm.backward(upstream);

        // numeric: objective = sum( softmax(logits) * upstream )
        Tensor numeric = finiteDiffGradScalarObjective(x -> softmaxDotObjective(x, upstream), logits, 1e-6);

        TestSupport.assertTensorAllClose(analytic, numeric, 1e-5);
    }

    /** Numerical grad of objective = sum(f(x)) */
    private static Tensor finiteDiffGradSum(Function<Tensor, Tensor> f, Tensor x, double eps) {
        Tensor grad = new Tensor(x.rows, x.cols);

        for (int r = 0; r < x.rows; r++) {
            for (int c = 0; c < x.cols; c++) {
                int i = r * x.cols + c;
                double old = x.data[i];

                x.data[i] = old + eps;
                double plus = f.apply(x).sum();

                x.data[i] = old - eps;
                double minus = f.apply(x).sum();

                x.data[i] = old;

                grad.data[i] = (plus - minus) / (2.0 * eps);
            }
        }
        return grad;
    }

    /** Numerical grad of any scalar objective J(x) */
    private static Tensor finiteDiffGradScalarObjective(Function<Tensor, Double> objective, Tensor x, double eps) {
        Tensor grad = new Tensor(x.rows, x.cols);

        for (int r = 0; r < x.rows; r++) {
            for (int c = 0; c < x.cols; c++) {
                int i = r * x.cols + c;
                double old = x.data[i];

                x.data[i] = old + eps;
                double plus = objective.apply(x);

                x.data[i] = old - eps;
                double minus = objective.apply(x);

                x.data[i] = old;

                grad.data[i] = (plus - minus) / (2.0 * eps);
            }
        }
        return grad;
    }

    private static double softmaxDotObjective(Tensor logits, Tensor upstream) {
        Softmax sm = new Softmax();
        Tensor p = sm.forward(logits);

        double s = 0.0;
        for (int r = 0; r < p.rows; r++) {
            for (int c = 0; c < p.cols; c++) {
                s += p.data[r * p.cols + c] * upstream.data[r * upstream.cols + c];
            }
        }
        return s;
    }
}

