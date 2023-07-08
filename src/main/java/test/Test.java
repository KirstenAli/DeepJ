package test;

import org.jbackprop.ann.activationfunctions.ActivationFunctions;
import org.jbackprop.ann.lossfunctions.LossFunctions;
import org.jbackprop.ann.NetworkBuilder;
import org.jbackprop.dataset.DataSet;
import org.jbackprop.dataset.Row;

import java.util.List;

public class Test{

    public static void main(String[] args){
        var rows = List.of(
                new Row(new double[]{0.0,0.0}, new double[]{0.0}),
                new Row(new double[]{1.0,0.0}, new double[]{1.0}),
                new Row(new double[]{0.0,1.0}, new double[]{1.0}),
                new Row(new double[]{1.0,1.0}, new double[]{0.0}));

        var dataset = new DataSet(2,1);
        dataset.addRows(rows);

        //example1(dataset);
        example2(dataset);
    }

    public static void example1(DataSet dataSet){
        var networkBuilder = new NetworkBuilder();

        var network = networkBuilder
                .architecture(3,2,1)
                .dataSet(dataSet)
                .build();

        network.learn();
    }

    public static void example2(DataSet dataSet){
        var networkBuilder = new NetworkBuilder();

        var network = networkBuilder
                .architecture(500,400,300,200,100,50,25,10,5,4,3,2,1)
                .dataSet(dataSet)
                .activationFunction(ActivationFunctions.TANH)
                .lossFunction(LossFunctions.MSE)
                .learningRate(0.1)
                .momentum(0.1)
                .desiredLoss(0.01) // Training stops
                .epochs(1000000000)
                .beforeEpoch(net ->
                        System.out.println("Current Epoch: " + net.getCurrentEpoch()))
                .afterEpoch(net ->
                        System.out.println("Error of Epoch: " + net.getLossOfEpoch() + "\n"))
                .build();

        network.learn();
    }
}
