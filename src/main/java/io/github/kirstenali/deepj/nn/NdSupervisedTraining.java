package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.loss.LossFunction;
import io.github.kirstenali.deepj.optimisers.ParameterOptimizer;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import io.github.kirstenali.deepj.training.Trainer;
import java.util.Objects;
import java.util.Random;

/** First-axis minibatching for images, sequences and matrices, using DeepJ's existing Trainer. */
public final class NdSupervisedTraining {
    private NdSupervisedTraining() {}

    public static Trainer trainer(Layer model, LossFunction loss, ParameterOptimizer optimizer,
                                  Tensor inputs, Tensor targets, long seed) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(loss, "loss");
        Objects.requireNonNull(optimizer, "optimizer");
        if (inputs.rank() < 1 || targets.rank() < 1 || inputs.size(0) < 1 || inputs.size(0) != targets.size(0)) {
            throw new IllegalArgumentException("Inputs and targets need the same positive first-axis sample count");
        }
        Random random = new Random(seed);
        return new Trainer(size -> step(model, loss, optimizer, inputs, targets, size, random));
    }

    private static Tensor[] batch(Tensor input, Tensor target, int size, Random random) {
        NeuralChecks.positive(size, "batch size");
        if (size >= input.size(0)) return new Tensor[]{input, target};
        int[] indices = new int[size];
        for (int index = 0; index < size; index++) { indices[index] = random.nextInt(input.size(0)); }
        return new Tensor[]{input.indexSelect(0, indices), target.indexSelect(0, indices)};
    }

    private static float step(Layer model, LossFunction loss, ParameterOptimizer optimizer,
                              Tensor inputs, Tensor targets, int size, Random random) {
        try (var scope = TensorScope.open()) {
            model.zeroGrad();
            Tensor[] batch = batch(inputs, targets, size, random);
            Tensor error = loss.forward(model.forward(batch[0]), batch[1]);
            if (error.requiresGrad()) error.backward();
            float value = error.item();
            optimizer.step(model.parameters());
            return value;
        }
    }
}
