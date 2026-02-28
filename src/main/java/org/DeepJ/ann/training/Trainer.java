package org.DeepJ.ann.training;

/**
 * A small, reusable training loop wrapper.
 *
 * <p>This library supports different model/data shapes (e.g. supervised Tensor->Tensor models,
 * and causal language models that operate on token ids). Rather than duplicating full trainers,
 * {@link Trainer} delegates a single training step to a pluggable {@link StepFunction}.
 */
public final class Trainer {

    @FunctionalInterface
    public interface StepFunction {
        /**
         * Runs one optimization step and returns the average loss for that step.
         */
        double trainStep(int batchSize);
    }

    private final StepFunction stepFn;

    public Trainer(StepFunction stepFn) {
        if (stepFn == null) throw new IllegalArgumentException("stepFn must not be null");
        this.stepFn = stepFn;
    }

    public double trainStep(int batchSize) {
        return stepFn.trainStep(batchSize);
    }

    /**
     * Train until maxSteps or until EMA loss goes below targetEmaLoss (if provided).
     */
    public TrainingResult train(
            int maxSteps,
            int batchSize,
            int logEvery,
            double emaBeta,
            Double targetEmaLoss
    ) {
        if (maxSteps <= 0) throw new IllegalArgumentException("maxSteps must be > 0");
        if (batchSize <= 0) throw new IllegalArgumentException("batchSize must be > 0");
        if (logEvery <= 0) throw new IllegalArgumentException("logEvery must be > 0");
        if (emaBeta <= 0.0 || emaBeta >= 1.0) throw new IllegalArgumentException("emaBeta must be in (0,1)");

        double ema = Double.NaN;
        int step;
        double lastLoss = Double.NaN;

        for (step = 0; step < maxSteps; step++) {
            lastLoss = trainStep(batchSize);
            ema = Double.isNaN(ema) ? lastLoss : (emaBeta * ema + (1.0 - emaBeta) * lastLoss);

            if (step % logEvery == 0) {
                System.out.printf("step=%d loss=%.6f ema=%.6f%n", step, lastLoss, ema);
            }

            if (targetEmaLoss != null && step > 50 && ema <= targetEmaLoss) {
                break;
            }
        }

        return new TrainingResult(step + 1, lastLoss, ema);
    }
}
