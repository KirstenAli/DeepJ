package org.DeepJ.transformer;

import java.util.Random;

public class SelfAttentionLayer {
    private final int dModel;
    private final Random rand = new Random();

    private Tensor Wq, Wk, Wv;
    private Tensor input, Q, K, V, attention;

    public SelfAttentionLayer(int dModel) {
        this.dModel = dModel;
        this.Wq = Tensor.random(dModel, dModel, rand);
        this.Wk = Tensor.random(dModel, dModel, rand);
        this.Wv = Tensor.random(dModel, dModel, rand);
    }

    public Tensor forward(Tensor input) {
        this.input = input;
        this.Q = input.matmul(Wq);
        this.K = input.matmul(Wk);
        this.V = input.matmul(Wv);
        Tensor scores = Q.matmul(K.transpose()).scale(1.0 / Math.sqrt(dModel));
        this.attention = Tensor.softmaxRows(scores);
        return attention.matmul(V);
    }

    public void backward(Tensor dL_dOutput, Tensor target, double learningRate) {
        Tensor dOutput = dL_dOutput;

        Tensor dV = attention.transpose().matmul(dOutput);
        Tensor dAttention = dOutput.matmul(V.transpose());
        Tensor dScores = Tensor.applySoftmaxBackward(dAttention, attention)
                .scale(1.0 / Math.sqrt(dModel));

        Tensor dQ = dScores.matmul(K);
        Tensor dK = dScores.transpose().matmul(Q);

        Tensor dWq = input.transpose().matmul(dQ);
        Tensor dWk = input.transpose().matmul(dK);
        Tensor dWv = input.transpose().matmul(dV);

        Wq = Wq.subtract(dWq.scale(learningRate));
        Wk = Wk.subtract(dWk.scale(learningRate));
        Wv = Wv.subtract(dWv.scale(learningRate));
    }
}
