package test;

import org.DeepJ.transformer.SelfAttentionLayer;
import org.DeepJ.transformer.Tensor;

public class TrainSelfAttention {
    public static void main(String[] args) {
        int dim = 4;
        var input = new Tensor(new double[][] {
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0}
        });
        var target = new Tensor(new double[][] {
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0}
        });

        var attn = new SelfAttentionLayer(dim);

        for (int epoch = 0; epoch < 100000; epoch++) {
            var output = attn.forward(input);
            var dLoss = output.subtract(target);
            if (epoch % 100 == 0) {
                System.out.printf("Epoch %d - Loss: %.6f\n", epoch, output.mseLoss(target));
            }
            attn.backward(dLoss, 0.05);
        }

        attn.forward(input).print("Final Output:");
    }
}
