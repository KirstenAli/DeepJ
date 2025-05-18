package org.DeepJ.transformer;

import java.util.Random;

public class SelfAttentionLayer {
    private final int dModel;

    private Tensor Wq, Wk, Wv;
    private Tensor input, Q, K, V, attention;

    public SelfAttentionLayer(int dModel) {
        this.dModel = dModel;
        Random rand = new Random();
        this.Wq = Tensor.random(dModel, dModel, rand);
        this.Wk = Tensor.random(dModel, dModel, rand);
        this.Wv = Tensor.random(dModel, dModel, rand);
    }

    public Tensor forward(Tensor input) {
        this.input = input;
        computeQKV(input);
        Tensor scores = computeAttention(Q, K);
        this.attention = Tensor.softmaxRows(scores);
        return attention.matmul(V);
    }

    private void computeQKV(Tensor x) {
        this.Q = x.matmul(Wq);
        this.K = x.matmul(Wk);
        this.V = x.matmul(Wv);
    }

    private Tensor computeAttention(Tensor Q, Tensor K) {
        return Q.matmul(K.transpose()).scale(1.0 / Math.sqrt(dModel));
    }

    public void backward(Tensor dL_dOutput, double learningRate) {
        Tensor dV = attention.transpose().matmul(dL_dOutput);
        Tensor dAttention = dL_dOutput.matmul(V.transpose());
        Tensor dScores = computeDScores(dAttention);

        Tensor dQ = dScores.matmul(K);
        Tensor dK = dScores.transpose().matmul(Q);

        computeGradients(dQ, dK, dV, learningRate);
    }

    private Tensor computeDScores(Tensor dAttention) {
        return Tensor.applySoftmaxBackward(dAttention, attention)
                .scale(1.0 / Math.sqrt(dModel));
    }

    private void computeGradients(Tensor dQ, Tensor dK, Tensor dV, double lr) {
        Tensor dWq = input.transpose().matmul(dQ);
        Tensor dWk = input.transpose().matmul(dK);
        Tensor dWv = input.transpose().matmul(dV);

        Wq = Wq.subtract(dWq.scale(lr));
        Wk = Wk.subtract(dWk.scale(lr));
        Wv = Wv.subtract(dWv.scale(lr));
    }
}
