
package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.loss.MSELoss;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

public class SupervisedTrainingTest {

    @Test
    void supervisedTrainer_runs_andLossDoesNotExplode_onTinyRegression() {
        // y = 2x on 4 samples
        Tensor x = Tensor.from2D(new double[][]{{0},{1},{2},{3}});
        Tensor y = Tensor.from2D(new double[][]{{0},{2},{4},{6}});

        Linear model = new Linear(1, 1, new Random(1));
        Trainer trainer = SupervisedTraining.trainer(
                model,
                new MSELoss(),
                AdamW.defaultAdamW(0.05),
                x, y,
                123
        );

        double l0 = trainer.trainStep(4);
        double l1 = trainer.trainStep(4);

        Assertions.assertTrue(Double.isFinite(l0));
        Assertions.assertTrue(Double.isFinite(l1));
        Assertions.assertTrue(l1 <= l0 * 1.1, "loss should not blow up after one step");
    }

    @Test
    void supervisedTrainer_keepsSampledInputTargetPairsAligned() {
        Tensor x = Tensor.from2D(new double[][]{{0}, {1}, {2}, {3}, {4}});
        Tensor y = Tensor.from2D(new double[][]{{0}, {1}, {2}, {3}, {4}});

        Layer identity = new Layer() {
            @Override
            public Tensor forward(Tensor input) { return input; }

            @Override
            public Tensor backward(Tensor gradOutput) { return gradOutput; }

            @Override
            public List<Parameter> parameters() { return List.of(); }
        };

        Trainer trainer = SupervisedTraining.trainer(
                identity,
                new MSELoss(),
                AdamW.defaultAdamW(0.01),
                x, y,
                123L
        );

        for (int i = 0; i < 20; i++) {
            Assertions.assertEquals(0.0, trainer.trainStep(1), 1e-12,
                    "aligned x/y sampling should keep identity-model loss at zero");
        }
    }
}
