package test;

import org.DeepJ.ann.Network;
import org.DeepJ.ann.NetworkBuilder;
import org.DeepJ.dataset.DataSet;
import org.DeepJ.transformer.*;

import static org.DeepJ.transformer.Tensor.flattenTensor;
import static org.DeepJ.transformer.Tensor.unflattenToTensor;

public class TransformerIntegrationTest {

    public static void main(String[] args) {

        Tensor input = new Tensor(new double[][] {
                {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1}
        });

        double[] target = new double[]{1.0};

        SelfAttentionLayer attn = new SelfAttentionLayer(3);
        attn.setLearningRate(0.05);

        LayerNorm layerNorm = new LayerNorm(3);

        DataSet dataSet = new DataSet(9, 1);
        Network network = new NetworkBuilder()
                .architecture(9, 6, 1)
                .dataSet(dataSet)
                .build();

        for (int epoch = 0; epoch < 10000; epoch++) {
            // Forward pass
            Tensor attentionOutput = attn.forward(input);
            Tensor normedOutput = layerNorm.forward(attentionOutput);
            double[] nnInput = flattenTensor(normedOutput);
            network.forward(nnInput);

            double predicted = network.getOutput()[0];
            double loss = network.calculateLoss();

            if (epoch % 100 == 0) {
                System.out.printf("Epoch %d: Loss = %.6f, Prediction = %.4f\n", epoch, loss, predicted);
            }

            // Backward pass
            network.backward(target);
            double[] gradInput = network.getInputGradient();
            Tensor gradTensor = unflattenToTensor(gradInput, 3, 3);
            Tensor dNorm = layerNorm.backward(gradTensor);
            attn.backward(dNorm);

            // Update weights
            layerNorm.updateWeights();
            attn.updateWeights();
            network.updateWeights();
        }

        Tensor finalAttn = attn.forward(input);
        Tensor finalNorm = layerNorm.forward(finalAttn);
        network.forward(flattenTensor(finalNorm));
        System.out.println("Final prediction: " + network.getOutput()[0]);
    }
}
