package test;

import org.DeepJ.ann.NetworkBuilder;
import org.DeepJ.dataset.DataSet;
import org.DeepJ.transformer.*;

import static org.DeepJ.transformer.Tensor.flattenTensor;
import static org.DeepJ.transformer.Tensor.unflattenToTensor;

public class TransformerNetworkIntegrationTest {

    public static void main(String[] args) {

        var input = new Tensor(new double[][] {
                {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1}
        });

        var target = new double[]{1.0};

        var attention = new SelfAttentionLayer(3);

        var dataSet = new DataSet(9, 1);
        var network = new NetworkBuilder()
                .architecture(9, 6, 1)
                .dataSet(dataSet)
                .build();

        for (int epoch = 0; epoch < 10000; epoch++) {

            var attentionOutput = attention.forward(input);
            var nnInput = flattenTensor(attentionOutput);
            network.forwardPass(nnInput);

            var predicted = network.getOutput()[0];
            var loss = network.calculateLossOfIteration();

            if (epoch % 100 == 0) {
                System.out.printf("Epoch %d: Loss = %.6f, Prediction = %.4f\n", epoch, loss, predicted);
            }

            network.backwardPass(target);

            var gradInput = network.getInputGradient();
            var gradTensor = unflattenToTensor(gradInput, 3, 3);
            attention.backward(gradTensor, 0.05);

            network.adjustWeights();
        }

        var finalAttn = attention.forward(input);
        network.forwardPass(flattenTensor(finalAttn));
        System.out.println("Final prediction: " + network.getOutput()[0]);
    }
}
