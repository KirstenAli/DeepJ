package org.jbackprop;

import lombok.Getter;
import lombok.Setter;

@Setter @Getter
public class GlobalParams {
    private Class<Neuron> neuronClass;
    private LossFunction lossFunction;
    private double learningRate;
}
