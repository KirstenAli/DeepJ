package org.jbackprop;

import java.util.List;

public class MSE implements LossFunction{
    @Override
    public double calculateLoss(double target, double actual){
        return calculateMSE(target, actual);
    }

    @Override
    public double dLoss(double target, double actual){
        return calculateMSEDerivative(target, actual);
    }

    @Override
    public double calculateSumError(Layer outputlayer,
                                    List<Double> target){
        double sumError =0;
        var neurons = outputlayer.getNeurons();

        for(int i=0; i<neurons.size(); i++){
            sumError+= neurons.get(i)
                    .calculateLoss(target.get(i));
        }

        return sumError;
    }

    public static double calculateMSE(double target, double actual){
        return Math.pow((target - actual), 2);
    }

    public static double calculateMSEDerivative(double target, double actual){
        return 2 * (target - actual);
    }
}
