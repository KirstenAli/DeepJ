package test;

import org.DeepJ.annv2.NeuralNetworkBuilder;
import org.DeepJ.annv2.layers.LayerNorm;
import org.DeepJ.annv2.layers.SelfAttentionLayer;
import org.DeepJ.annv2.layers.ActivationLayer;
import org.DeepJ.annv2.layers.DenseLayer;
import org.DeepJ.annv2.layers.FlattenLayer;
import org.DeepJ.annv2.NeuralNetwork;
import org.DeepJ.annv2.activations.Tanh;
import org.DeepJ.annv2.loss.MSELoss;
import org.DeepJ.annv2.optimisers.OptimizerFactory;
import org.DeepJ.annv2.optimisers.SGDMomentum;
import org.DeepJ.annv2.Tensor;

public class ANNv2Test {
    public static void main(String[] args) {
        // === [1] Input ===
        Tensor input = new Tensor(new double[][]{
                {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1}
        });

        Tensor target = new Tensor(new double[][]{
                {1.0}
        });

        // === [2] Build & Train Model ===
        OptimizerFactory opt = () -> new SGDMomentum(0.1, 0.1);

        NeuralNetwork net = new NeuralNetworkBuilder()
                .input(input)
                .target(target)
                .loss(new MSELoss())
                .epochs(10000)
                .targetLoss(0.001)
                .learningRate(0.1)
                .logLoss(true)
                .addLayer(new SelfAttentionLayer(3))
                .addLayer(new LayerNorm(3))
                .addLayer(new FlattenLayer())
                .addLayer(new DenseLayer(9, 6, opt))
                .addLayer(new ActivationLayer(new Tanh()))
                .addLayer(new DenseLayer(6, 1, opt))
                .addLayer(new ActivationLayer(new Tanh()))
                .build();

        net.train();

        // === [3] Predict ===
        Tensor pred = net.forward(input);
        pred.print("Prediction:");
        target.print("Target:");
    }
}