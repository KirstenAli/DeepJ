package org.DeepJ.annv2.layers;

import org.DeepJ.annv2.Tensor;

import java.util.Random;

public class SelfAttentionLayer implements Layer {
    private final int dModel;
    private final boolean mask;

    private Tensor Wq, Wk, Wv;
    private Tensor dQ, dK, dV;
    private Tensor dWq, dWk, dWv;
    private Tensor input, Q, K, V, attention;
    private double learningRate;

    public SelfAttentionLayer(int dModel) {
        this(dModel, true);
    }

    public SelfAttentionLayer(int dModel, boolean mask) {
        this.dModel = dModel;
        this.mask = mask;
        Random rand = new Random();
        this.Wq = Tensor.random(dModel, dModel, rand);
        this.Wk = Tensor.random(dModel, dModel, rand);
        this.Wv = Tensor.random(dModel, dModel, rand);
    }

    public Tensor forward(Tensor input) {
        this.input = input;
        computeQKV(input);
        Tensor scores = computeAttention(Q, K);
        if (mask) scores = applyMask(scores);
        this.attention = Tensor.softmaxRows(scores);
        return attention.matmul(V);
    }

    private Tensor applyMask(Tensor scores) {
        return scores.add(Tensor.causalMask(scores.rows));
    }

    private void computeQKV(Tensor x) {
        this.Q = x.matmul(Wq);
        this.K = x.matmul(Wk);
        this.V = x.matmul(Wv);
    }

    private Tensor computeAttention(Tensor Q, Tensor K) {
        return Q.matmul(K.transpose()).multiplyScalar(1.0 / Math.sqrt(dModel));
    }

    public Tensor backward(Tensor dL_dOutput, double learningRate) {
        this.learningRate = learningRate;

        dV = attention.transpose().matmul(dL_dOutput);
        Tensor dAttention = dL_dOutput.matmul(V.transpose());
        Tensor dScores = computeDScores(dAttention);

        dQ = dScores.matmul(K);
        dK = dScores.transpose().matmul(Q);

        computeGradients();
        return computeInputGradient();
    }

    private Tensor computeDScores(Tensor dAttention) {
        return Tensor.softmaxBackward(dAttention, attention)
                .multiplyScalar(1.0 / Math.sqrt(dModel));
    }

    private void computeGradients() {
        dWq = input.transpose().matmul(dQ);
        dWk = input.transpose().matmul(dK);
        dWv = input.transpose().matmul(dV);
    }

    private Tensor computeInputGradient() {
        Tensor dInputQ = dQ.matmul(Wq.transpose());
        Tensor dInputK = dK.matmul(Wk.transpose());
        Tensor dInputV = dV.matmul(Wv.transpose());
        return dInputQ.add(dInputK).add(dInputV);
    }

    public void step() {
        Wq = Wq.subtract(dWq.multiplyScalar(learningRate));
        Wk = Wk.subtract(dWk.multiplyScalar(learningRate));
        Wv = Wv.subtract(dWv.multiplyScalar(learningRate));
    }

    public int getdModel() {
        return dModel;
    }

    public Tensor getWq() {
        return Wq;
    }

    public Tensor getWk() {
        return Wk;
    }

    public Tensor getWv() {
        return Wv;
    }

    public Tensor getdQ() {
        return dQ;
    }

    public Tensor getdK() {
        return dK;
    }

    public Tensor getdV() {
        return dV;
    }

    public Tensor getdWq() {
        return dWq;
    }

    public Tensor getdWk() {
        return dWk;
    }

    public Tensor getdWv() {
        return dWv;
    }

    public Tensor getInput() {
        return input;
    }

    public Tensor getQ() {
        return Q;
    }

    public Tensor getK() {
        return K;
    }

    public Tensor getV() {
        return V;
    }

    public Tensor getAttention() {
        return attention;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}
