package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.activations.ActivationFunction;
import io.github.kirstenali.deepj.optimisers.Parameter;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

public final class FNN implements Layer {

    private final List<Linear> linears = new ArrayList<>();
    private final List<ActivationFunction> activations = new ArrayList<>();
    private final ActivationFunction outputActivation;

    private final List<Tensor> hiddenPreActs = new ArrayList<>();

    public FNN(
            int inputSize,
            int[] hiddenSizes,
            int outputSize,
            Supplier<ActivationFunction> hiddenActivationFactory,
            ActivationFunction outputActivation,
            Random rnd
    ) {
        validateArguments(inputSize, hiddenSizes, outputSize, hiddenActivationFactory, rnd);
        int lastSize = buildHiddenLayers(inputSize, hiddenSizes, hiddenActivationFactory, rnd);
        linears.add(new Linear(lastSize, outputSize, rnd));
        this.outputActivation = outputActivation;
    }

    private static void validateArguments(int inputSize, int[] hiddenSizes, int outputSize,
                                          Supplier<ActivationFunction> activationFactory, Random rnd) {
        requirePositive(inputSize, "inputSize");
        requirePositive(outputSize, "outputSize");
        if (hiddenSizes == null) throw new IllegalArgumentException("hiddenSizes must not be null");
        if (rnd == null) throw new IllegalArgumentException("rnd must not be null");
        if (hiddenSizes.length > 0 && activationFactory == null) throw missingActivationFactory();
    }

    private int buildHiddenLayers(int inputSize, int[] sizes,
                                  Supplier<ActivationFunction> activationFactory, Random rnd) {
        int currentSize = inputSize;
        for (int size : sizes) {
            requirePositive(size, "hidden layer size");
            linears.add(new Linear(currentSize, size, rnd));
            activations.add(activationFactory.get());
            currentSize = size;
        }
        return currentSize;
    }

    private static void requirePositive(int value, String name) {
        if (value <= 0) throw new IllegalArgumentException(name + " must be > 0");
    }

    private static IllegalArgumentException missingActivationFactory() {
        return new IllegalArgumentException(
                "hiddenActivationFactory must not be null when hiddenSizes is non-empty");
    }

    public FNN(int inputSize, int[] hiddenSizes, int outputSize,
               Supplier<ActivationFunction> hiddenActivationFactory, Random rnd) {
        this(inputSize, hiddenSizes, outputSize, hiddenActivationFactory, null, rnd);
    }

    @Override
    public Tensor forward(Tensor x) {
        hiddenPreActs.clear();

        Tensor h = x;
        int hiddenCount = activations.size();

        for (int i = 0; i < hiddenCount; i++) {
            Tensor z = linears.get(i).forward(h);
            hiddenPreActs.add(z);
            h = activations.get(i).forward(z);
        }

        Tensor out = linears.get(linears.size() - 1).forward(h);
        if (outputActivation != null) {
            out = outputActivation.forward(out);
        }
        return out;
    }

    @Override
    public Tensor backward(Tensor gradOut) {
        Tensor g = gradOut;

        if (outputActivation != null) {
            g = outputActivation.backward(g);
        }

        g = linears.get(linears.size() - 1).backward(g);

        for (int i = activations.size() - 1; i >= 0; i--) {
            g = activations.get(i).backward(g);
            g = linears.get(i).backward(g);
        }
        return g;
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        for (Linear lin : linears) {
            ps.addAll(lin.parameters());
        }
        return ps;
    }
}
