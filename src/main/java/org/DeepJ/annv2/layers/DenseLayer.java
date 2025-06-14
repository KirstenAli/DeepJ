package org.DeepJ.annv2.layers;

import org.DeepJ.annv2.optimisers.Optimizer;
import org.DeepJ.annv2.optimisers.OptimizerFactory;
import org.DeepJ.annv2.Tensor;

import java.util.Random;

public class DenseLayer implements Layer {

    private Tensor weights;
    private Tensor biases;
    private Tensor input;

    private Tensor gradW;
    private Tensor gradB;

    private Optimizer wOpt;
    private Optimizer bOpt;

    private double learningRate;

    public DenseLayer(int inSize, int outSize) {
        this(inSize, outSize, null);
    }

    public DenseLayer(int inSize, int outSize, OptimizerFactory factory) {
        Random rand = new Random();
        double std = Math.sqrt(1.0 / inSize);
        this.weights = Tensor.random(inSize, outSize, rand).multiplyScalar(std / 0.1);
        this.biases = Tensor.zeros(1, outSize);

        if (factory != null) {
            this.wOpt = factory.create();
            this.bOpt = factory.create();
        }
    }

    @Override
    public Tensor forward(Tensor in) {
        this.input = in;
        return in.matmul(weights).addBroadcastRows(biases);
    }

    @Override
    public Tensor backward(Tensor gradOut, double learningRate) {
        this.learningRate = learningRate;

        Tensor gradIn = gradOut.matmul(weights.transpose());

        this.gradW = input.transpose().matmul(gradOut);
        this.gradB = gradOut.sumAlongCols();

        return gradIn;
    }

    public void step() {
        if (wOpt != null) {
            weights = wOpt.apply(weights, gradW);
            biases = bOpt.apply(biases, gradB);
        } else {
            weights = weights.subtract(gradW.multiplyScalar(learningRate));
            biases = biases.subtract(gradB.multiplyScalar(learningRate));
        }
    }

    public Tensor getWeights() { return weights; }
    public Tensor getBiases() { return biases; }
    public Tensor getGradW() { return gradW; }
    public Tensor getGradB() { return gradB; }
}
