package test;

import org.DeepJ.ann.ActivationLayer;
import org.DeepJ.ann.DenseLayer;
import org.DeepJ.ann.NeuralNetwork;
import org.DeepJ.ann.activations.Tanh;
import org.DeepJ.ann.loss.MSELoss;
import org.DeepJ.ann.optimisers.OptimizerFactory;
import org.DeepJ.ann.optimisers.SGDMomentum;
import org.DeepJ.transformer.Tensor;

public class ANNv2 {

    public static void main(String[] args) {
        double[][] in  = {{0,0}, {0,1}, {1,0}, {1,1}};
        double[][] out = {{0},   {1},   {1},   {0}};
        Tensor input  = new Tensor(in);
        Tensor target = new Tensor(out);

        OptimizerFactory opt = (r, c) -> new SGDMomentum(0.1, 0.1);

        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(2, 3, opt));
        net.addLayer(new ActivationLayer(new Tanh()));

        net.addLayer(new DenseLayer(3, 2, opt));
        net.addLayer(new ActivationLayer(new Tanh()));

        net.addLayer(new DenseLayer(2, 1, opt));
        net.addLayer(new ActivationLayer(new Tanh()));

        int    epochs = 10000;
        double lr     = 0.1;

        System.out.println("Training network on XOR with MSE loss…");
        net.train(input, target, new MSELoss(), epochs, lr);

        Tensor pred = net.forward(input);
        System.out.println();
        pred.print("Predictions:");
        target.print("Targets:");
    }
}