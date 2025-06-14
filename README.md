# DeepJ
DeepJ is an object-oriented artificial neural network (ANN) library for Java.

[View the Docs](https://kirstenali.github.io/DeepJ/)

Getting Started:

```java
 public static void main(String[] args) {
    // === [1] Input ===
    Tensor input = new Tensor(new double[][]{
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1}
    });

    Tensor target = new Tensor(new double[][]{
            {1.0}
    });

    // === [2] Build & Train Model ===
    OptimizerFactory opt = () -> new SGDMomentum(0.1, 0.1);

    NeuralNetwork net = new NeuralNetworkBuilder()
            .input(input)
            .target(target)
            .loss(new MSELoss())
            .epochs(10000)
            .learningRate(0.1)
            .logLoss(true)
            .addLayer(new SelfAttentionLayer(3))
            .addLayer(new LayerNorm(3))
            .addLayer(new FlattenLayer())
            .addLayer(new DenseLayer(9, 6, opt))
            .addLayer(new ActivationLayer(new Tanh()))
            .addLayer(new DenseLayer(6, 1, opt))
            .addLayer(new ActivationLayer(new Tanh()))
            .build();

    net.train();

    // === [3] Predict ===
    Tensor pred = net.forward(input);
    pred.print("Prediction:");
    target.print("Target:");
}
```
