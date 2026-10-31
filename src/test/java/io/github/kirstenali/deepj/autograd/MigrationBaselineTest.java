package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.activations.Tanh;
import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.data.BatchSource;
import io.github.kirstenali.deepj.layers.FNN;
import io.github.kirstenali.deepj.loss.MSELoss;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.models.origin.DeepJOrigin;
import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.models.orbit.DeepJOrbit;
import io.github.kirstenali.deepj.models.orbit.DeepJOrbitConfig;
import io.github.kirstenali.deepj.models.prism.DeepJPrism;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.models.nova.DeepJNova;
import io.github.kirstenali.deepj.models.nova.DeepJNovaConfig;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import io.github.kirstenali.deepj.training.SupervisedTraining;
import io.github.kirstenali.deepj.training.Trainer;
import org.junit.jupiter.api.Test;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/** Fixture captured using the unmodified manual-backprop master commit 374f8464. */
class MigrationBaselineTest {
    @Test
    void gradientsAndThreeAdamUpdatesMatchAllFiveManualTrainingPaths() throws IOException {
        Tensor.setBackend(new CpuBackend());
        var resource = getClass().getResourceAsStream("/autograd/manual-baseline.bin");
        assertNotNull(resource);
        try (var expected = new DataInputStream(resource)) {
            for (DecoderOnlyModel model : models()) {
                checkModel(expected, model);
            }
            checkFeedForward(expected);
            assertEquals(-1, expected.read(), "fixture must be fully consumed");
        }
    }

    private static List<DecoderOnlyModel> models() {
        return List.of(new DeepJOrigin(new DeepJOriginConfig(7, 3, 4, 2, 1, 6), 42),
                new DeepJOrbit(new DeepJOrbitConfig(7, 3, 4, 2, 1, 6), 42),
                new DeepJPrism(new DeepJPrismConfig(7, 3, 4, 2, 1, 6, 3, 2), 42),
                new DeepJNova(new DeepJNovaConfig(7, 3, 8, 4, 2, 1, 6), 42));
    }

    private static void checkModel(DataInputStream expected, DecoderOnlyModel model) throws IOException {
        Tensor prediction = model.forward(new int[]{1, 3, 5});
        checkTensor(expected, prediction);
        prediction.backward(Tensor.random(3, 7, new Random(99)));
        checkGradients(expected, model.parameters());
        BatchSource source = size -> new Batch(new int[][]{{1, 3, 5}, {5, 2, 1}},
                new int[][]{{3, 5, 2}, {2, 1, 4}});
        checkUpdates(expected, CausalLMTraining.trainer(model, source, 0.001f), model.parameters());
    }

    private static void checkFeedForward(DataInputStream expected) throws IOException {
        var model = new FNN(2, new int[]{3}, 2, Tanh::new, new Random(42));
        Tensor x = Tensor.from2D(new float[][]{{0.2f, -0.3f}, {0.4f, 0.1f}});
        Tensor y = Tensor.from2D(new float[][]{{0.1f, 0.2f}, {-0.2f, 0.3f}});
        Tensor prediction = model.forward(x);
        checkTensor(expected, prediction);
        prediction.backward(Tensor.random(2, 2, new Random(99)));
        checkGradients(expected, model.parameters());
        Trainer trainer = SupervisedTraining.trainer(model, new MSELoss(),
                AdamW.defaultAdamW(0.001f), x, y, 42);
        checkUpdates(expected, trainer, model.parameters());
    }

    private static void checkGradients(DataInputStream expected, List<Parameter> parameters) throws IOException {
        for (Parameter parameter : parameters) {
            checkTensor(expected, parameter.grad());
        }
    }

    private static void checkUpdates(DataInputStream expected, Trainer trainer,
                                      List<Parameter> parameters) throws IOException {
        for (int step = 0; step < 3; step++) {
            assertEquals(expected.readFloat(), trainer.trainStep(2), 2e-6f, "training loss");
            for (Parameter parameter : parameters) {
                checkTensor(expected, parameter.value);
            }
        }
    }

    private static void checkTensor(DataInputStream expected, Tensor actual) throws IOException {
        actual.materialize();
        for (float value : actual.rawData()) {
            float reference = expected.readFloat();
            assertEquals(reference, value, 2e-5f + Math.abs(reference) * 3e-4f);
        }
    }
}
