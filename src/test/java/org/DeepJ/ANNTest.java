package org.DeepJ;

import org.DeepJ.ann.NeuralNetwork;
import org.DeepJ.ann.NeuralNetworkBuilder;
import org.DeepJ.ann.Tensor;
import org.DeepJ.ann.activations.Tanh;
import org.DeepJ.ann.layers.*;
import org.DeepJ.ann.loss.MSELoss;
import org.DeepJ.ann.optimisers.OptimizerFactory;
import org.DeepJ.ann.optimisers.SGDMomentum;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class ANNTest {

    @Test
    public void testNetworkLearnsOneHotToOneMapping() {
        // Input
        Tensor input = new Tensor(new double[][]{
                {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1}
        });

        // Target
        Tensor target = new Tensor(new double[][]{
                {0, 0, 1}
        });

        // Optimizer
        OptimizerFactory opt = () -> new SGDMomentum(0.1, 0.1);

        // Build network
        NeuralNetwork net = new NeuralNetworkBuilder()
                .input(input)
                .target(target)
                .loss(new MSELoss())
                .epochs(10_000)
                .targetLoss(0.001)
                .learningRate(0.1)
                .logLoss(true)
                .addLayer(new SelfAttentionLayer(3))
                .addLayer(new LayerNorm(3))
                .addLayer(new FlattenLayer())
                .addLayer(new DenseLayer(9, 6, opt))
                .addLayer(new ActivationLayer(new Tanh()))
                .addLayer(new DenseLayer(6, 3, opt))
                .addLayer(new ActivationLayer(new Tanh()))
                .build();

        net.train();

        // Forward pass to get prediction
        Tensor pred = net.forward(input);

        // Check prediction against target
        double loss = pred.mseLoss(target);
        System.out.printf("Final loss: %.6f%n", loss);

        // Assert that the final loss is below threshold
        assertTrue(loss < 0.01, "Model failed to learn the mapping (loss too high)");

        // print predictions
        System.out.println("Prediction:");
        pred.print("Output:");
    }
}
