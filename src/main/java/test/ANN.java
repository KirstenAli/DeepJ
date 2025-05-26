package test;

import org.DeepJ.ann_oop.activationfunctions.ActivationFunctions;
import org.DeepJ.ann_oop.lossfunctions.LossFunctions;
import org.DeepJ.ann_oop.NetworkBuilder;
import org.DeepJ.dataset.DataSet;
import org.DeepJ.persistence.PersistenceManager;

public class ANN {

    public static void main(String[] args){

        var dataSet = new DataSet(2,1);

        dataSet.addRow(new double[]{0.0, 0.0}, new double[]{0.0});
        dataSet.addRow(new double[]{1.0, 0.0}, new double[]{1.0});
        dataSet.addRow(new double[]{0.0, 1.0}, new double[]{1.0});
        dataSet.addRow(new double[]{1.0, 1.0}, new double[]{0.0});

        //example1(dataSet);
        example2(dataSet);
    }

    public static void example1(DataSet dataSet){
        var networkBuilder = new NetworkBuilder();

        var network = networkBuilder
                .architecture(3,2,1)
                .dataSet(dataSet)
                .build();

        network.learn();

        network.save("my_network");

        network = PersistenceManager.loadNetwork("my_network");

        network.saveWeightsAsJsonFile("network_weights");
    }

    public static void example2(DataSet dataSet){
        var networkBuilder = new NetworkBuilder();

        var network = networkBuilder
                .architecture(3,2,1)
                .dataSet(dataSet)
                .activationFunction(ActivationFunctions.TANH)
                .lossFunction(LossFunctions.MSE)
                .learningRate(0.1)
                .momentum(0.1)
                .desiredLoss(0.01) // Training stops
                .epochs(1000000000) // Training stops
                .beforeEpoch(net ->
                        System.out.println("Current Epoch:" + net.getCurrentEpoch()))
                .afterEpoch(net ->
                        System.out.println("Loss of Epoch:" + net.getLossOfEpoch() + "\n"))
                .build();

        network.learn();
    }
}
