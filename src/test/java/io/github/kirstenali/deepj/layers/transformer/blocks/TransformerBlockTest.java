package io.github.kirstenali.deepj.layers.transformer.blocks;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.loss.MSELoss;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class TransformerBlockTest {

    @Test
    void forward_isIdentity_whenAllTrainableParametersAreZero() {
        int dModel = 4;
        DeepJOriginTransformerBlock block = new DeepJOriginTransformerBlock(dModel, 2, 8, new Random(1));

        for (Parameter p : block.parameters()) {
            p.value = Tensor.zeros(p.value.rows, p.value.cols);
            p.zeroGrad();
        }

        Tensor x = Tensor.from2D(new float[][]{
                { 1,  2,  3,  4},
                {-1, -2, -3, -4}
        });

        Tensor y = block.forward(x);
        TestSupport.assertTensorAllClose(x, y, 1e-12f);
    }

    @Test
    void backward_returnsSameShape_andAccumulatesSomeGradients() {
        DeepJOriginTransformerBlock block = new DeepJOriginTransformerBlock(4, 2, 8, new Random(2));

        Tensor x = Tensor.from2D(new float[][]{
                { 0.2f, -0.1f,  0.3f,  0.0f},
                { 0.0f,  0.4f, -0.2f,  0.1f},
                {-0.3f,  0.2f,  0.1f, -0.4f}
        });

        Tensor y = block.forward(x);
        Tensor gradOut = Tensor.ones(y.rows, y.cols);
        Tensor gradIn = block.backward(gradOut);

        TestSupport.assertTensorShape(gradIn, x.rows, x.cols);

        double totalGrad = 0.0f;
        for (Parameter p : block.parameters()) {
            totalGrad += p.grad.sumAbs();
        }
        assertTrue(totalGrad > 0.0f, "Expected some non-zero gradients in block parameters");
    }

    @Test
    void orbitBlockRejectsInvalidHeadDimensionsCleanly() {
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOrbitTransformerBlock(4, 0, 8, 16, new Random(1L)));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOrbitTransformerBlock(5, 2, 8, 16, new Random(1L)));
    }

    @Test
    void learning_can_reduce_mse_loss_within_a_few_steps() {
        DeepJOriginTransformerBlock block = new DeepJOriginTransformerBlock(4, 2, 8, new Random(3));
        AdamW opt = new AdamW(0.01f, 0.9f, 0.999f, 1e-8f, 0.0f);

        Tensor x = Tensor.from2D(new float[][]{
                { 0.2f, -0.1f,  0.3f,  0.0f},
                { 0.0f,  0.4f, -0.2f,  0.1f},
                {-0.3f,  0.2f,  0.1f, -0.4f}
        });

        Tensor target = Tensor.zeros(x.rows, x.cols);

        TestSupport.assertLossDecreases(() -> trainOneStepMSE(block, opt, x, target), 10,
                "Expected MSE loss to decrease within a few optimizer steps");
    }

    private static double trainOneStepMSE(DeepJOriginTransformerBlock block, AdamW opt, Tensor x, Tensor target) {
        Tensor y = block.forward(x);

        MSELoss mse = new MSELoss();
        double loss = mse.loss(y, target);
        Tensor gradOut = mse.gradient(y, target);

        block.backward(gradOut);
        opt.step(block.parameters());
        for (Parameter p : block.parameters()) {
            p.zeroGrad();
        }

        return loss;
    }
}
