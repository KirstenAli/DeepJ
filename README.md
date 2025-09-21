# DeepJ
DeepJ is an object-oriented artificial neural network (ANN) library for Java.

[View the Docs](https://kirstenali.github.io/DeepJ/)

🚀 Installation  
To get started, add the following dependency to your `pom.xml`:

```xml
<repositories>
    <repository>
        <id>github</id>
        <url>https://maven.pkg.github.com/KirstenAli/DeepJ</url>
    </repository>
</repositories>

<dependencies>
    <dependency>
        <groupId>io.github.kirstenali</groupId>
        <artifactId>deepj</artifactId>
        <version>0.1.0-alpha</version>
    </dependency>
</dependencies>
```

📚 Examples

```java
public static void main(String[] args) {
    // Input: 3 tokens (one-hot encoded)
    Tensor input = new Tensor(new double[][]{
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1}
    });

    // Target: one-hot output vector
    Tensor target = new Tensor(new double[][]{
            {0, 0, 1}
    });

    // Optimizer
    OptimizerFactory opt = () -> new SGDMomentum(0.1, 0.1);

    // Build model
    NeuralNetwork net = new NeuralNetworkBuilder()
            .input(input)
            .target(target)
            .loss(new MSELoss())
            .epochs(10_000)
            .targetLoss(0.001)
            .learningRate(0.1)
            .logLoss(true)
            .addLayer(new SelfAttentionLayer(3))
            .addLayer(new LayerNorm(3))
            .addLayer(new FlattenLayer())
            .addLayer(new DenseLayer(9, 6, opt))
            .addLayer(new ActivationLayer(new Tanh()))
            .addLayer(new DenseLayer(6, 3, opt))
            .addLayer(new ActivationLayer(new Tanh()))
            .build();

    // Train
    net.train();

    // Predict and print
    Tensor pred = net.forward(input);
    pred.print("Prediction:");
    target.print("Target:");
}
```
