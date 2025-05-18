package test;

import org.DeepJ.ann.Network;
import org.DeepJ.ann.NetworkBuilder;
import org.DeepJ.dataset.DataSet;
import org.DeepJ.transformer.*;

import static org.DeepJ.transformer.Tensor.flattenTensor;
import static org.DeepJ.transformer.Tensor.unflattenToTensor;

public class TransformerNetworkIntegrationTest {

    public static void main(String[] args) {

        Tensor input = new Tensor(new double[][] {
                {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1}
        });

        double[] target = new double[]{1.0};

        SelfAttentionLayer attention = new SelfAttentionLayer(3, 0.05);

        DataSet dataSet = new DataSet(9, 1);
        Network network = new NetworkBuilder()
                .architecture(9, 6, 1)
                .dataSet(dataSet)
                .build();

        for (int epoch = 0; epoch < 10000; epoch++) {
            Tensor attentionOutput = attention.forward(input);
            double[] nnInput = flattenTensor(attentionOutput);
            network.forward(nnInput);

            double predicted = network.getOutput()[0];
            double loss = network.calculateLoss();

            if (epoch % 100 == 0) {
                System.out.printf("Epoch %d: Loss = %.6f, Prediction = %.4f\n", epoch, loss, predicted);
            }

            network.backward(target);

            double[] gradInput = network.getInputGradient();
            Tensor gradTensor = unflattenToTensor(gradInput, 3, 3);
            attention.backward(gradTensor);

            attention.adjustWeights();
            network.adjustWeights();
        }

        Tensor finalAttn = attention.forward(input);
        network.forward(flattenTensor(finalAttn));
        System.out.println("Final prediction: " + network.getOutput()[0]);
    }
}
