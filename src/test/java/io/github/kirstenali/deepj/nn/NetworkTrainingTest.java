package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.activations.ReLU;
import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.loss.MSELoss;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.training.Trainer;
import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.Random;
import static io.github.kirstenali.deepj.nn.NeuralTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

class NetworkTrainingTest extends NeuralTest {
    @Test void cnnAndEveryRecurrentFamilyLearnUsingTheExistingAdamWAndAutograd() {
        List<Module> networks = List.of(new Sequential(new Conv2D(1, 2, 1, new Random(1)),
                Activation.tanh(), new GlobalAveragePool(), new Linear(2, 1, new Random(2))),
                new RNN(2, 2, new Random(3)), new GRU(2, 2, new Random(4)), new LSTM(2, 2, new Random(5)));
        Tensor images = values(3, 1, 2, 2), sequence = values(3, 2, 2);
        train(networks.get(0), images, Tensor.ones(3, 1).multiplyScalar(0.3f));
        for (int index = 1; index < networks.size(); index++) {
            train(networks.get(index), sequence, Tensor.ones(6, 2).reshape(3, 2, 2).multiplyScalar(0.3f));
        }
    }

    @Test void denseAutoencoderAndEncoderTrainThroughTheSameSupervisedPath() {
        Module autoencoder = new Autoencoder(new FNN(2, new int[0], 1, ReLU::new, new Random(1)),
                new FNN(1, new int[0], 2, ReLU::new, new Random(2)));
        train(autoencoder, values(3, 2), values(3, 2));
        TransformerEncoder encoder = new TransformerEncoder(new TransformerEncoderConfig(2, 1, 3, 1, 0), new Random(1));
        train(encoder, values(3, 2, 2), Tensor.ones(6, 2).reshape(3, 2, 2).multiplyScalar(0.1f));
    }

    @Test void minibatchesSelectMatchingFirstAxisSamplesAndRejectInvalidArguments() {
        Linear model = new Linear(2, 1, new Random(1));
        Tensor inputs = values(5, 2), targets = values(5, 1);
        Trainer trainer = NdSupervisedTraining.trainer(model, new MSELoss(), AdamW.defaultAdamW(0.01f),
                inputs, targets, 42);
        assertTrue(Float.isFinite(trainer.trainStep(2)));
        assertThrows(IllegalArgumentException.class, () -> trainer.trainStep(0));
        assertThrows(IllegalArgumentException.class, () -> NdSupervisedTraining.trainer(model, new MSELoss(),
                AdamW.defaultAdamW(0.01f), inputs, values(4, 1), 1));
        assertThrows(IllegalArgumentException.class, () -> NdSupervisedTraining.trainer(model, new MSELoss(),
                AdamW.defaultAdamW(0.01f), new Tensor(), values(1), 1));
        assertThrows(IllegalArgumentException.class, () -> NdSupervisedTraining.trainer(model, new MSELoss(),
                AdamW.defaultAdamW(0.01f), values(1), new Tensor(), 1));
    }

    private static void train(Module model, Tensor inputs, Tensor targets) {
        Trainer trainer = NdSupervisedTraining.trainer(model, new MSELoss(), AdamW.defaultAdamW(0.03f),
                inputs, targets, 7);
        float initial = trainer.trainStep(inputs.size(0)), last = initial;
        for (int step = 0; step < 35; step++) { last = trainer.trainStep(inputs.size(0)); }
        assertTrue(last < initial * 0.6f, model.getClass().getSimpleName() + ": " + initial + " -> " + last);
    }
}
