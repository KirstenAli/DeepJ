package org.DeepJ;

import org.DeepJ.ann.Tensor;
import org.DeepJ.ann.layers.SelfAttentionLayer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class TrainSelfAttentionTest {

    @Test
    public void testSelfAttentionLearnsIdentityMapping() {
        int dim = 4;

        Tensor input = new Tensor(new double[][] {
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0}
        });

        Tensor target = new Tensor(new double[][] {
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0}
        });

        SelfAttentionLayer attn = new SelfAttentionLayer(dim);

        double learningRate = 0.05;
        int maxEpochs = 5000;
        double targetLoss = 1e-4;

        double loss = Double.MAX_VALUE;

        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            Tensor output = attn.forward(input);
            Tensor dLoss = output.subtract(target);
            loss = output.mseLoss(target);

            if (epoch % 500 == 0) {
                System.out.printf("Epoch %d - Loss: %.6f%n", epoch, loss);
            }

            attn.backward(dLoss, learningRate);
            attn.step();

            if (loss < targetLoss) break;
        }

        System.out.printf("Final loss: %.6f%n", loss);

        assertTrue(loss < targetLoss, "Self-attention layer failed to learn identity mapping");
    }
}
