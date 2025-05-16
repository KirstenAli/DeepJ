package test;

import org.DeepJ.transformer.SelfAttentionLayer;
import org.DeepJ.transformer.Tensor;

public class TrainSelfAttention {
    public static void main(String[] args) {
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

        for (int epoch = 0; epoch < 1000; epoch++) {
            Tensor output = attn.forward(input);
            Tensor dLoss = output.subtract(target);
            if (epoch % 100 == 0) {
                System.out.printf("Epoch %d - Loss: %.6f\n", epoch, output.mseLoss(target));
            }
            attn.backward(dLoss, target, 0.05);
        }

        attn.forward(input).print("Final Output:");
    }
}
