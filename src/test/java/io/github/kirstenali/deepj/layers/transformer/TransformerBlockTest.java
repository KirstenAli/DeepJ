package io.github.kirstenali.deepj.layers.transformer;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.optimisers.Parameter;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class TransformerBlockTest {

    @Test
    void forward_isIdentity_whenAllTrainableParametersAreZero() {
        int dModel = 4;
        TransformerBlock block = new TransformerBlock(dModel, 2, 8, new Random(1));

        // Zero ALL trainable params. With residual connections, this should yield y == x.
        for (Parameter p : block.parameters()) {
            p.value = Tensor.zeros(p.value.rows, p.value.cols);
            p.zeroGrad();
        }

        Tensor x = TestSupport.tensor(new double[][]{
                { 1,  2,  3,  4},
                {-1, -2, -3, -4}
        });

        Tensor y = block.forward(x);
        TestSupport.assertTensorAllClose(x, y, 1e-12);
    }

    @Test
    void backward_returnsSameShape_andAccumulatesSomeGradients() {
        TransformerBlock block = new TransformerBlock(4, 2, 8, new Random(2));

        Tensor x = TestSupport.tensor(new double[][]{
                { 0.2, -0.1,  0.3,  0.0},
                { 0.0,  0.4, -0.2,  0.1},
                {-0.3,  0.2,  0.1, -0.4}
        });

        Tensor y = block.forward(x);
        Tensor gradOut = Tensor.ones(y.rows, y.cols);
        Tensor gradIn = block.backward(gradOut);

        TestSupport.assertTensorShape(gradIn, x.rows, x.cols);

        // Not every param is guaranteed to be non-zero, but at least one should be.
        double totalGrad = 0.0;
        for (Parameter p : block.parameters()) totalGrad += TestSupport.tensorSumAbs(p.grad);
        assertTrue(totalGrad > 0.0, "Expected some non-zero gradients in block parameters");
    }
}
