package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.activations.ActivationFunction;
import io.github.kirstenali.deepj.optimisers.Parameter;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

/**
 * Flexible fully-connected neural network (MLP) built from {@link Linear} projections.
 *
 * <p>This class is intentionally small: it exists as a convenience for users who want a
 * classic ANN-style model while deepj remains transformer-oriented.
 *
 * <p>Design notes:
 * <ul>
 *   <li>Uses {@link Linear} so parameters can be optimized externally (AdamW/SGD/etc.).</li>
 *   <li>Accepts an {@link ActivationFunction} factory to avoid state-sharing bugs during backprop.</li>
 * </ul>
 */
public final class FNN implements Layer {

    private final List<Linear> linears = new ArrayList<>();
    private final List<ActivationFunction> activations = new ArrayList<>();
    private final ActivationFunction outputActivation; // may be null

    // Cache last pre-activations per hidden layer (for clarity; activations cache internally too)
    private final List<Tensor> hiddenPreActs = new ArrayList<>();

    /**
     * Build an MLP of the form:
     * Linear -> act -> Linear -> act -> ... -> Linear -> (optional outputAct)
     */
    public FNN(
            int inputSize,
            int[] hiddenSizes,
            int outputSize,
            Supplier<ActivationFunction> hiddenActivationFactory,
            ActivationFunction outputActivation,
            Random rnd
    ) {
        if (inputSize <= 0) throw new IllegalArgumentException("inputSize must be > 0");
        if (outputSize <= 0) throw new IllegalArgumentException("outputSize must be > 0");
        if (hiddenSizes == null) throw new IllegalArgumentException("hiddenSizes must not be null");
        if (rnd == null) throw new IllegalArgumentException("rnd must not be null");
        if (hiddenSizes.length > 0 && hiddenActivationFactory == null) {
            throw new IllegalArgumentException("hiddenActivationFactory must not be null when hiddenSizes is non-empty");
        }

        int in = inputSize;
        for (int h : hiddenSizes) {
            if (h <= 0) throw new IllegalArgumentException("hidden layer size must be > 0");
            linears.add(new Linear(in, h, rnd));
            activations.add(hiddenActivationFactory.get());
            in = h;
        }
        // output projection
        linears.add(new Linear(in, outputSize, rnd));
        this.outputActivation = outputActivation;
    }

    public FNN(int inputSize, int[] hiddenSizes, int outputSize, Supplier<ActivationFunction> hiddenActivationFactory, Random rnd) {
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

        // final linear
        Tensor out = linears.get(linears.size() - 1).forward(h);
        if (outputActivation != null) {
            out = outputActivation.forward(out);
        }
        return out;
    }

    public Tensor backward(Tensor gradOut) {
        Tensor g = gradOut;

        if (outputActivation != null) {
            g = outputActivation.backward(g);
        }

        // final linear grad
        g = linears.get(linears.size() - 1).backward(g);

        // hidden layers (reverse)
        for (int i = activations.size() - 1; i >= 0; i--) {
            g = activations.get(i).backward(g);
            g = linears.get(i).backward(g);
        }
        return g;
    }

    @Override
    public Tensor backward(Tensor gradOutput, double learningRate) {
        return backward(gradOutput);
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        for (Linear lin : linears) ps.addAll(lin.parameters());
        return ps;
    }
}
