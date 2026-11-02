package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.activations.Tanh;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.nn.*;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.List;
import java.util.Random;

final class ModelCases {
    private ModelCases() {}
    static List<String> tensorFamilies() {
        return List.of("fnn", "conv1d", "conv2d", "rnn", "gru", "lstm", "autoencoder", "encoder", "cnn", "pool");
    }

    static Layer tensor(String family, DType dtype, long seed) {
        Random random = new Random(seed);
        ParameterFactory parameters = new ParameterFactory(dtype);
        return switch (family) {
            case "fnn" -> new FNN(2, new int[]{3}, 1, Tanh::new, null, random, parameters);
            case "autoencoder" -> autoencoder(random, parameters);
            case "encoder" -> new TransformerEncoder(new TransformerEncoderConfig(4, 2, 6, 1, 0), random, parameters);
            case "pool" -> Pool1D.average(2);
            case "batchnorm" -> new Sequential(new BatchNorm(2, 1e-5f, 0.1f, parameters.scope("norm")),
                    new Linear(2, 1, random, parameters.scope("head")));
            default -> spatialOrRecurrent(family, random, parameters);
        };
    }

    private static Layer spatialOrRecurrent(String family, Random random, ParameterFactory parameters) {
        return switch (family) {
            case "conv1d" -> new Conv1D(1, 2, 2, 1, 0, 1, true, random, parameters);
            case "conv2d" -> new Conv2D(1, 2, window(), true, random, parameters);
            case "cnn" -> cnn(random, parameters);
            default -> recurrent(family, random, parameters);
        };
    }

    private static Layer recurrent(String family, Random random, ParameterFactory parameters) {
        return switch (family) {
            case "rnn" -> new RNN(2, 2, random, parameters);
            case "gru" -> new GRU(2, 2, random, parameters);
            case "lstm" -> new LSTM(2, 2, random, parameters);
            default -> throw new IllegalArgumentException(family);
        };
    }

    private static Window2D window() { return new Window2D(2, 2, 1, 1, 0, 0, 1, 1); }

    private static Layer autoencoder(Random random, ParameterFactory parameters) {
        return new Autoencoder(new Linear(2, 1, random, parameters.scope("encoder")),
                new Linear(1, 2, random, parameters.scope("decoder")));
    }

    private static Layer cnn(Random random, ParameterFactory parameters) {
        return new Sequential(new Conv2D(1, 2, window(), true, random, parameters.scope("conv")),
                Activation.tanh(), Pool2D.average(2), new Flatten(),
                new Linear(2, 1, random, parameters.scope("head")));
    }

    static TensorBatch data(String family) {
        int[] input = inputShape(family), target = targetShape(family);
        return new TensorBatch(values(input), values(target).multiplyScalar(0.3f));
    }

    private static int[] inputShape(String family) {
        return switch (family) {
            case "conv1d", "pool" -> new int[]{7, 1, 4};
            case "conv2d", "cnn" -> new int[]{7, 1, 3, 3};
            case "rnn", "gru", "lstm" -> new int[]{7, 3, 2};
            case "encoder" -> new int[]{7, 3, 4};
            default -> new int[]{7, 2};
        };
    }

    private static int[] targetShape(String family) {
        return switch (family) {
            case "fnn", "cnn", "batchnorm" -> new int[]{7, 1};
            case "conv1d" -> new int[]{7, 2, 3};
            case "conv2d" -> new int[]{7, 2, 2, 2};
            case "pool" -> new int[]{7, 1, 2};
            default -> inputShape(family);
        };
    }

    static Tensor values(int... shape) {
        int count = java.util.Arrays.stream(shape).reduce(1, Math::multiplyExact);
        float[] values = new float[count];
        for (int index = 0; index < count; index++) { values[index] = (index % 11 - 5) / 13f; }
        return Tensor.from(values, shape);
    }
}
