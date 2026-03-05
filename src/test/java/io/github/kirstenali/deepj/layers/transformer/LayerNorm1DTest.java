package io.github.kirstenali.deepj.layers.transformer;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class LayerNorm1DTest {

    @Test
    void forward_producesZeroMeanUnitVariancePerRow_whenGammaOneBetaZero() {
        LayerNorm1D ln = new LayerNorm1D(4);

        Tensor x = TestSupport.tensor(new double[][]{
                { 1.0,  2.0,  3.0,  4.0},
                {-2.0,  0.0,  2.0,  4.0},
                {10.0, 10.0, 10.0, 10.0} // constant row -> should become ~0s due to EPS
        });

        Tensor y = ln.forward(x);
        TestSupport.assertTensorShape(y, 3, 4);

        for (int r = 0; r < y.rows; r++) {
            double mean = TestSupport.rowMean(y, r);
            assertEquals(0.0, mean, 1e-7, "row mean should be ~0");

            double var = TestSupport.rowVar(y, r);
            // constant row variance tends to 0 (not 1). For non-constant, should be ~1.
            if (r != 2) {
                assertEquals(1.0, var, 1e-4, "row variance should be ~1");
            } else {
                assertTrue(var < 1e-6, "constant row should normalize to ~0 variance");
            }
        }
    }

    @Test
    void backward_accumulatesGammaAndBetaGradients_andReturnsInputShape() {
        LayerNorm1D ln = new LayerNorm1D(3);

        Tensor x = TestSupport.tensor(new double[][]{
                {1.0, 2.0, 3.0},
                {2.0, 0.0, -2.0}
        });
        Tensor y = ln.forward(x);

        Tensor gradOut = Tensor.ones(y.rows, y.cols);
        Tensor gradIn = ln.backward(gradOut);

        TestSupport.assertTensorShape(gradIn, 2, 3);

        var params = ln.parameters();
        assertEquals(2, params.size());
        assertTrue(TestSupport.tensorSumAbs(params.get(0).grad) > 0.0, "gamma.grad should be non-zero");
        assertTrue(TestSupport.tensorSumAbs(params.get(1).grad) > 0.0, "beta.grad should be non-zero");
    }
}
