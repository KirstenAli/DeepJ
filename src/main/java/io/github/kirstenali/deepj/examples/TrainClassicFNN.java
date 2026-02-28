package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.Tensor;
import io.github.kirstenali.deepj.activations.GELU;
import io.github.kirstenali.deepj.layers.FNN;
import io.github.kirstenali.deepj.loss.MSELoss;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.training.SupervisedTraining;
import io.github.kirstenali.deepj.training.Trainer;
import io.github.kirstenali.deepj.training.TrainingResult;

import java.util.Random;

/**
 * Example: classic MLP training using Linear + activation (via FNN) and the unified Trainer wrapper.
 */
public final class TrainClassicFNN {

    public static void main(String[] args) {
        // Tiny regression toy: learn to map one-hot -> target vector
        Tensor x = new Tensor(new double[][]{
                {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1}
        });

        Tensor y = new Tensor(new double[][]{
                {0, 0, 1},
                {0, 1, 0},
                {1, 0, 0}
        });

        Random rnd = new Random(42);
        FNN mlp = new FNN(
                3,
                new int[]{16, 16},
                3,
                GELU::new,
                null,
                rnd
        );

        MSELoss lossFn = new MSELoss();
        AdamW opt = AdamW.defaultAdamW(3e-3);

        Trainer trainer = SupervisedTraining.trainer(mlp, lossFn, opt, x, y, 123L);

        TrainingResult res = trainer.train(
                3000,   // maxSteps
                3,      // batchSize (full batch here)
                200,    // logEvery
                0.98,   // emaBeta
                1e-6    // targetEmaLoss (stop if effectively solved)
        );

        System.out.printf("%nFinished at step=%d lastLoss=%.6f ema=%.6f%n",
                res.steps(), res.lastLoss(), res.emaLoss());

        System.out.println("\nFinal prediction:");
        System.out.println(mlp.forward(x));
    }
}
