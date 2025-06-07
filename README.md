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

public static void transformerIntegrationTest() {
    Tensor input = new Tensor(new double[][] {
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1}
    });

    double[] target = new double[]{1.0};

    SelfAttentionLayer attn = new SelfAttentionLayer(3);
    attn.setLearningRate(0.05);

    LayerNorm layerNorm = new LayerNorm(3);

    DataSet dataSet = new DataSet(9, 1);
    Network network = new NetworkBuilder()
            .architecture(9, 6, 1)
            .dataSet(dataSet)
            .build();

    for (int epoch = 0; epoch < 10000; epoch++) {
        // Forward pass
        Tensor attentionOutput = attn.forward(input);
        Tensor normedOutput = layerNorm.forward(attentionOutput);
        double[] nnInput = flattenTensor(normedOutput);
        network.forward(nnInput);

        double predicted = network.getOutput()[0];
        double loss = network.calculateLoss();

        if (epoch % 100 == 0) {
            System.out.printf("Epoch %d: Loss = %.6f, Prediction = %.4f\n", epoch, loss, predicted);
        }

        // Backward pass
        network.backward(target);
        double[] gradInput = network.getInputGradient();
        Tensor gradTensor = unflattenToTensor(gradInput, 3, 3);
        Tensor dNorm = layerNorm.backward(gradTensor);
        attn.backward(dNorm);

        // Update weights
        layerNorm.updateWeights();
        attn.updateWeights();
        network.updateWeights();
    }

    Tensor finalAttn = attn.forward(input);
    Tensor finalNorm = layerNorm.forward(finalAttn);
    network.forward(flattenTensor(finalNorm));
    System.out.println("Final prediction: " + network.getOutput()[0]);
}

//transformer.ann (experimental)
public static void main(String[] args) {
    double[][] in  = {{0,0}, {0,1}, {1,0}, {1,1}};
    double[][] out = {{0},   {1},   {1},   {0}};
    Tensor input  = new Tensor(in);
    Tensor target = new Tensor(out);

    OptimizerFactory opt = (r, c) -> new SGDMomentum(0.1, 0.1);

    NeuralNetwork net = new NeuralNetwork();
    net.addLayer(new DenseLayer(2, 3, opt));
    net.addLayer(new ActivationLayer(new Tanh()));

    net.addLayer(new DenseLayer(3, 2, opt));
    net.addLayer(new ActivationLayer(new Tanh()));

    net.addLayer(new DenseLayer(2, 1, opt));
    net.addLayer(new ActivationLayer(new Tanh()));

    int    epochs = 10000;
    double lr     = 0.1;

    System.out.println("Training network on XOR with MSE loss…");
    net.train(input, target, new MSELoss(), epochs, lr);

    Tensor pred = net.forward(input);
    System.out.println();
    pred.print("Predictions:");
    target.print("Targets:");
}

```
