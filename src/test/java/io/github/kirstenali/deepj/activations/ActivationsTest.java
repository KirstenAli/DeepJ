package io.github.kirstenali.deepj.activations;

import io.github.kirstenali.deepj.testing.AutogradTestSupport;
import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

public class ActivationsTest {

    @Test
    void relu_forwardBackward() {
        ReLU relu = new ReLU();
        Tensor y = relu.forward(reluInput());
        TestSupport.assertTensorAllClose(y, reluOutput(), 1e-12f);
        Tensor gradOut = Tensor.from2D(new float[][]{{1, 1, 1}, {2, 2, 2}});
        Tensor gx = AutogradTestSupport.backward(relu::forward, reluInput(), gradOut);
        TestSupport.assertTensorAllClose(gx, reluGradient(), 1e-12f);
    }

    private static Tensor reluInput() {
        return Tensor.from2D(new float[][]{{-1, 0, 2}, {3, -4, 5}});
    }

    private static Tensor reluOutput() {
        return Tensor.from2D(new float[][]{{0, 0, 2}, {3, 0, 5}});
    }

    private static Tensor reluGradient() {
        return Tensor.from2D(new float[][]{{0, 0, 1}, {2, 0, 2}});
    }

    @Test
    void sigmoid_outputsInRange_andBackwardNonNegativeForPositiveUpstream() {
        Sigmoid s = new Sigmoid();
        Tensor x = Tensor.from2D(new float[][]{{-10, 0, 10}});
        Tensor y = s.forward(x);

        for (int c = 0; c < y.matrixCols(); c++) {
            Assertions.assertTrue(y.rawData()[c] > 0.0f && y.rawData()[c] < 1.0f);
        }

        Tensor gradOut = Tensor.from2D(new float[][]{{1, 1, 1}});
        Tensor gx = AutogradTestSupport.backward(s::forward, x, gradOut);
        for (int c = 0; c < gx.matrixCols(); c++) {
            Assertions.assertTrue(gx.rawData()[c] >= 0.0f, "sigmoid' should be >= 0");
        }
    }

    @Test
    void tanh_isOddFunctionApprox() {
        Tanh t = new Tanh();
        Tensor x1 = Tensor.from2D(new float[][]{{0.5f, -0.5f}});
        Tensor y1 = t.forward(x1);
        Assertions.assertEquals(y1.rawData()[0], -y1.rawData()[1], 1e-12f);
    }

    @Test
    void gelu_isSmooth_andMonotonicAroundZero() {
        GELU g = new GELU();
        Tensor x = Tensor.from2D(new float[][]{{-1e-3f, 0.0f, 1e-3f}});
        Tensor y = g.forward(x);

        Assertions.assertTrue(y.rawData()[0] < y.rawData()[1]);
        Assertions.assertTrue(y.rawData()[1] < y.rawData()[2]);
    }

    @Test
    void softmax_rowsSumTo1() {
        Softmax sm = new Softmax();
        Tensor logits = Tensor.from2D(new float[][]{
                {1, 2, 3},
                {-1, 0, 1}
        });
        Tensor p = sm.forward(logits);

        for (int r = 0; r < p.matrixRows(); r++) {
            float sum = 0.0f;
            for (int c = 0; c < p.matrixCols(); c++) {
                sum += p.rawData()[r * p.matrixCols() + c];
            }
            Assertions.assertEquals(1.0f, sum, 1e-6f);
        }

        Assertions.assertThrows(IllegalStateException.class, () -> p.backward(p));
    }

    @Test
    void sigmoid_backward_matchesFiniteDifference() {
        Tensor x = Tensor.from2D(new float[][]{{-1.2f, 0.0f, 2.3f}});

        Sigmoid s = new Sigmoid();
        Tensor y = s.forward(x);
        Tensor analytic = AutogradTestSupport.backward(s::forward, x, Tensor.ones(y.matrixRows(), y.matrixCols()));

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
        Tensor analytic = AutogradTestSupport.backward(t::forward, x, Tensor.ones(y.matrixRows(), y.matrixCols()));

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
        Tensor analytic = AutogradTestSupport.backward(g::forward, x, Tensor.ones(y.matrixRows(), y.matrixCols()));

        Tensor numeric = finiteDiffGradSum(u -> {
            GELU gg = new GELU();
            return gg.forward(u);
        }, x, 1e-3f);

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

        Softmax sm = new Softmax();
        sm.forward(logits);
        Tensor analytic = AutogradTestSupport.backward(sm::forward, logits, upstream);

        Tensor numeric = finiteDiffGradScalarObjective(x -> softmaxDotObjective(x, upstream), logits, 1e-3f);

        TestSupport.assertTensorAllClose(analytic, numeric, 2e-3f);
    }

    private static Tensor finiteDiffGradSum(Function<Tensor, Tensor> f, Tensor x, float eps) {
        Tensor grad = new Tensor(x.matrixRows(), x.matrixCols());

        for (int r = 0; r < x.matrixRows(); r++) {
            for (int c = 0; c < x.matrixCols(); c++) {
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

    private static Tensor finiteDiffGradScalarObjective(Function<Tensor, Float> objective, Tensor x, float eps) {
        Tensor grad = new Tensor(x.matrixRows(), x.matrixCols());

        for (int r = 0; r < x.matrixRows(); r++) {
            for (int c = 0; c < x.matrixCols(); c++) {
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
        for (int r = 0; r < p.matrixRows(); r++) {
            for (int c = 0; c < p.matrixCols(); c++) {
                s += p.rawData()[r * p.matrixCols() + c] * upstream.rawData()[r * upstream.matrixCols() + c];
            }
        }
        return s;
    }
}
