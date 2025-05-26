package org.DeepJ.ann;

import org.DeepJ.ann.optimisers.Optimizer;
import org.DeepJ.ann.optimisers.OptimizerFactory;
import org.DeepJ.transformer.Tensor;

import java.util.Random;

public class DenseLayer implements Layer {

    private Tensor weights;
    private Tensor biases;
    private Tensor input;

    private Optimizer wOpt;
    private Optimizer bOpt;

    public DenseLayer(int inSize, int outSize) {
        this(inSize, outSize, null);
    }

    public DenseLayer(int inSize, int outSize, OptimizerFactory factory) {
        Random rand = new Random();
        double std = Math.sqrt(1.0 / inSize);
        this.weights = Tensor.random(inSize, outSize, rand).multiplyScalar(std / 0.1);
        this.biases = Tensor.zeros(1, outSize);

        if (factory != null) {
            this.wOpt = factory.create(weights.rows, weights.cols);
            this.bOpt = factory.create(biases.rows, biases.cols);
        }
    }

    @Override
    public Tensor forward(Tensor in) {
        this.input = in;
        return in.matmul(weights).addBroadcastRows(biases);
    }

    @Override
    public Tensor backward(Tensor gradOut, double learningRate) {
        Tensor gradIn = gradOut.matmul(weights.transpose());
        Tensor gradW = input.transpose().matmul(gradOut);
        Tensor gradB = gradOut.sumAlongCols();

        if (wOpt != null) {
            weights = wOpt.apply(weights, gradW);
            biases = bOpt.apply(biases, gradB);
        } else {
            weights = weights.subtract(gradW.multiplyScalar(learningRate));
            biases = biases.subtract(gradB.multiplyScalar(learningRate));
        }
        return gradIn;
    }
}