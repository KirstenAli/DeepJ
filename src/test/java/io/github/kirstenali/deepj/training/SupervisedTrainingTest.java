
package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.loss.MSELoss;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Random;

public class SupervisedTrainingTest {

    @Test
    void supervisedTrainer_runs_andLossDoesNotExplode_onTinyRegression() {
        // y = 2x on 4 samples
        Tensor x = new Tensor(new double[][]{{0},{1},{2},{3}});
        Tensor y = new Tensor(new double[][]{{0},{2},{4},{6}});

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
}
