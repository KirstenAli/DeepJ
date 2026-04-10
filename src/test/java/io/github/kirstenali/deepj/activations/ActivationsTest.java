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
        Tensor x = Tensor.from2D(new float[][]{
                {-1, 0, 2},
                {3, -4, 5}
        });

        Tensor y = relu.forward(x);
        TestSupport.assertTensorAllClose(y, Tensor.from2D(new float[][]{
                {0, 0, 2},
                {3, 0, 5}
        }), 1e-12f);

        Tensor gradOut = Tensor.from2D(new float[][]{
                {1, 1, 1},
                {2, 2, 2}
        });
        Tensor gx = relu.backward(gradOut);
        TestSupport.assertTensorAllClose(gx, Tensor.from2D(new float[][]{
                {0, 0, 1},
                {2, 0, 2}
        }), 1e-12f);
    }

    @Test
    void sigmoid_outputsInRange_andBackwardNonNegativeForPositiveUpstream() {
        Sigmoid s = new Sigmoid();
        Tensor x = Tensor.from2D(new float[][]{{-10, 0, 10}});
        Tensor y = s.forward(x);

        for (int c = 0; c < y.cols; c++) {
            Assertions.assertTrue(y.data[c] > 0.0f && y.data[c] < 1.0f);
        }

        Tensor gradOut = Tensor.from2D(new float[][]{{1, 1, 1}});
        Tensor gx = s.backward(gradOut);
        for (int c = 0; c < gx.cols; c++) {
            Assertions.assertTrue(gx.data[c] >= 0.0f, "sigmoid' should be >= 0");
        }
    }

    @Test
    void tanh_isOddFunctionApprox() {
        Tanh t = new Tanh();
        Tensor x1 = Tensor.from2D(new float[][]{{0.5f, -0.5f}});
        Tensor y1 = t.forward(x1);
        Assertions.assertEquals(y1.data[0], -y1.data[1], 1e-12f);
    }

    @Test
    void gelu_isSmooth_andMonotonicAroundZero() {
        GELU g = new GELU();
        Tensor x = Tensor.from2D(new float[][]{{-1e-3f, 0.0f, 1e-3f}});
        Tensor y = g.forward(x);

        Assertions.assertTrue(y.data[0] < y.data[1]);
        Assertions.assertTrue(y.data[1] < y.data[2]);
    }

    @Test
    void softmax_rowsSumTo1() {
        Softmax sm = new Softmax();
        Tensor logits = Tensor.from2D(new float[][]{
                {1, 2, 3},
                {-1, 0, 1}
        });
        Tensor p = sm.forward(logits);

        for (int r = 0; r < p.rows; r++) {
            float sum = 0.0f;
            for (int c = 0; c < p.cols; c++) sum += p.data[r * p.cols + c];
            Assertions.assertEquals(1.0f, sum, 1e-6f);
        }

        // backward should require forward
        Softmax sm2 = new Softmax();
        Assertions.assertThrows(IllegalStateException.class, () -> sm2.backward(p));
    }

    @Test
    void sigmoid_backward_matchesFiniteDifference() {
        Tensor x = Tensor.from2D(new float[][]{{-1.2f, 0.0f, 2.3f}});

        // analytic: d/dx sum(sigmoid(x)) = sigmoid'(x)
        Sigmoid s = new Sigmoid();
        Tensor y = s.forward(x);
        Tensor analytic = s.backward(Tensor.ones(y.rows, y.cols));

        // numeric
        Tensor numeric = finiteDiffGradSum(t -> {
            Sigmoid ss = new Sigmoid();
            return ss.forward(t);
        }, x, 1e-3f);

        TestSupport.assertTensorAllClose(analytic, numeric, 1e-3f);
    }

    @Test
    void tanh_backward_matchesFiniteDifference() {
        Tensor x = Tensor.from2D(new float[][]{{-0.7f, 0.2f, 1.1f}});

        Tanh t = new Tanh();
        Tensor y = t.forward(x);
        Tensor analytic = t.backward(Tensor.ones(y.rows, y.cols));

        Tensor numeric = finiteDiffGradSum(u -> {
            Tanh tt = new Tanh();
            return tt.forward(u);
        }, x, 1e-3f);

        TestSupport.assertTensorAllClose(analytic, numeric, 1e-3f);
    }

    @Test
    void gelu_backward_matchesFiniteDifference() {
        Tensor x = Tensor.from2D(new float[][]{{-1.5f, -0.2f, 0.0f, 0.4f, 2.0f}});

        GELU g = new GELU();
        Tensor y = g.forward(x);
        Tensor analytic = g.backward(Tensor.ones(y.rows, y.cols));

        Tensor numeric = finiteDiffGradSum(u -> {
            GELU gg = new GELU();
            return gg.forward(u);
        }, x, 1e-3f);

        // GELU is approximate + exp/tanh internally -> slightly looser tolerance
        TestSupport.assertTensorAllClose(analytic, numeric, 2e-3f);
    }

    @Test
    void softmax_backward_matchesFiniteDifference_forDotObjective() {
        Tensor logits = Tensor.from2D(new float[][]{
                { 0.3f, -1.2f, 2.0f },
                { 1.0f,  0.5f, -0.7f }
        });

        Tensor upstream = Tensor.from2D(new float[][]{
                { 0.7f, -0.2f, 1.3f },
                { -1.1f, 0.4f, 0.9f }
        });

        // analytic: grad = softmax.backward(upstream)
        Softmax sm = new Softmax();
        sm.forward(logits);
        Tensor analytic = sm.backward(upstream);

        // numeric: objective = sum( softmax(logits) * upstream )
        Tensor numeric = finiteDiffGradScalarObjective(x -> softmaxDotObjective(x, upstream), logits, 1e-3f);

        TestSupport.assertTensorAllClose(analytic, numeric, 2e-3f);
    }

    /** Numerical grad of objective = sum(f(x)) */
    private static Tensor finiteDiffGradSum(Function<Tensor, Tensor> f, Tensor x, float eps) {
        Tensor grad = new Tensor(x.rows, x.cols);

        for (int r = 0; r < x.rows; r++) {
            for (int c = 0; c < x.cols; c++) {
                float old = x.get(r, c);

                x.set(r, c, old + eps);
                float plus = f.apply(x).sum();

                x.set(r, c, old - eps);
                float minus = f.apply(x).sum();

                x.set(r, c, old);

                grad.set(r, c, (plus - minus) / (2.0f * eps));
            }
        }
        return grad;
    }

    /** Numerical grad of any scalar objective J(x) */
    private static Tensor finiteDiffGradScalarObjective(Function<Tensor, Float> objective, Tensor x, float eps) {
        Tensor grad = new Tensor(x.rows, x.cols);

        for (int r = 0; r < x.rows; r++) {
            for (int c = 0; c < x.cols; c++) {
                float old = x.get(r, c);

                x.set(r, c, old + eps);
                float plus = objective.apply(x);

                x.set(r, c, old - eps);
                float minus = objective.apply(x);

                x.set(r, c, old);

                grad.set(r, c, (plus - minus) / (2.0f * eps));
            }
        }
        return grad;
    }

    private static float softmaxDotObjective(Tensor logits, Tensor upstream) {
        Softmax sm = new Softmax();
        Tensor p = sm.forward(logits);

        float s = 0.0f;
        for (int r = 0; r < p.rows; r++) {
            for (int c = 0; c < p.cols; c++) {
                s += p.data[r * p.cols + c] * upstream.data[r * upstream.cols + c];
            }
        }
        return s;
    }
}

