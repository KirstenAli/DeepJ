package org.DeepJ.annoop.lossfunctions;

public class MSE extends LossFunction {
    @Override
    public double calculateActualLoss(double target, double actual){
        return target-actual;
    }

    @Override
    public double calculateLoss(double actualLoss){
        return actualLoss*actualLoss;
    }

    @Override
    public double derivative(double actualLoss){
        return actualLoss; //precisely 2*actualLoss
    }
}
