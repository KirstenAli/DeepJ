# DeepJ
DeepJ is an object-oriented artificial neural network (ANN) library for Java.

[View the Docs](https://kirstenali.github.io/DeepJ/)

Getting Started:

```java
 public static void main(String[] args) {

    var dataSet = new DataSet(2, 1);

    dataSet.addRow(new double[]{0.0, 0.0}, new double[]{0.0});
    dataSet.addRow(new double[]{1.0, 0.0}, new double[]{1.0});
    dataSet.addRow(new double[]{0.0, 1.0}, new double[]{1.0});
    dataSet.addRow(new double[]{1.0, 1.0}, new double[]{0.0});

    example1(dataSet);
    example2(dataSet);
}

public static void example1(DataSet dataSet) {
    var networkBuilder = new NetworkBuilder();

    var network = networkBuilder
            .architecture(3, 2, 1)
            .dataSet(dataSet)
            .build();

    network.learn();

    network.save("my_network");

    network = PersistenceManager.loadNetwork("my_network");

    network.saveWeightsAsJsonFile("network_weights");
}

public static void example2(DataSet dataSet) {
    var networkBuilder = new NetworkBuilder();

    var network = networkBuilder
            .architecture(500, 400, 300, 200, 100, 50, 25, 10, 5, 4, 3, 2, 1)
            .dataSet(dataSet)
            .activationFunction(ActivationFunctions.TANH)
            .lossFunction(LossFunctions.MSE)
            .learningRate(0.1)
            .momentum(0.1)
            .desiredLoss(0.01)
            .epochs(1000000000)
            .beforeEpoch(net ->
                    System.out.println("Current Epoch:" + net.getCurrentEpoch()))
            .afterEpoch(net ->
                    System.out.println("Loss of Epoch:" + net.getLossOfEpoch() + "\n"))
            .build();

    network.learn();
}

//SelfAttentionLayer NN integration test
public static void example3() {
    Tensor input = new Tensor(new double[][] {
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1}
    });

    double[] target = new double[]{1.0};

    SelfAttentionLayer attention = new SelfAttentionLayer(3);

    DataSet dataSet = new DataSet(9, 1);
    Network network = new NetworkBuilder()
            .architecture(9, 6, 1)
            .dataSet(dataSet)
            .build();

    for (int epoch = 0; epoch < 10000; epoch++) {
        Tensor attentionOutput = attention.forward(input);
        double[] nnInput = flattenTensor(attentionOutput);
        network.forwardPass(nnInput);

        double predicted = network.getOutput()[0];
        double loss = network.calculateLoss();

        if (epoch % 100 == 0) {
            System.out.printf("Epoch %d: Loss = %.6f, Prediction = %.4f\n", epoch, loss, predicted);
        }

        network.backwardPass(target);

        double[] gradInput = network.getInputGradient();
        Tensor gradTensor = unflattenToTensor(gradInput, 3, 3);
        attention.backward(gradTensor, 0.05);

        network.adjustWeights();
    }

    Tensor finalAttn = attention.forward(input);
    network.forwardPass(flattenTensor(finalAttn));
    System.out.println("Final prediction: " + network.getOutput()[0]);
}

```
