package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.tensor.Tensor;

/**
 * A small, reusable training loop wrapper.
 *
 * <p>This library supports different model/data shapes (e.g. supervised Tensor->Tensor models,
 * and causal language models that operate on token ids). Rather than duplicating full trainers,
 * {@link Trainer} delegates a single training step to a pluggable {@link StepFunction}.
 */
public final class Trainer {

    private static final int DEFAULT_RELEASE_EVERY_STEPS = 25;

    @FunctionalInterface
    public interface StepFunction {
        /**
         * Runs one optimization step and returns the average loss for that step.
         */
        double trainStep(int batchSize);
    }

    @FunctionalInterface
    public interface StepHook {
        void onStep(int step, double loss, double ema) throws Exception;
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
     * Uses the default periodic backend release cadence.
     */
    public TrainingResult train(
            int maxSteps,
            int batchSize,
            int logEvery,
            double emaBeta,
            Double targetEmaLoss
    ) {
        return train(maxSteps, batchSize, logEvery, emaBeta, targetEmaLoss, DEFAULT_RELEASE_EVERY_STEPS, null);
    }

    public TrainingResult train(
            int maxSteps,
            int batchSize,
            int logEvery,
            double emaBeta,
            Double targetEmaLoss,
            int releaseEverySteps
    ) {
        return train(maxSteps, batchSize, logEvery, emaBeta, targetEmaLoss, releaseEverySteps, null);
    }

    /**
     * Train until maxSteps or until EMA loss goes below targetEmaLoss (if provided).
     * Uses the default periodic backend release cadence.
     */
    public TrainingResult train(
            int maxSteps,
            int batchSize,
            int logEvery,
            double emaBeta,
            Double targetEmaLoss,
            StepHook stepHook
    ) {
        return train(maxSteps, batchSize, logEvery, emaBeta, targetEmaLoss, DEFAULT_RELEASE_EVERY_STEPS, stepHook);
    }

    /**
     * Train until maxSteps or until EMA loss goes below targetEmaLoss (if provided).
     * releaseEverySteps <= 0 disables periodic release, but final release still runs.
     */
    public TrainingResult train(
            int maxSteps,
            int batchSize,
            int logEvery,
            double emaBeta,
            Double targetEmaLoss,
            int releaseEverySteps,
            StepHook stepHook
    ) {
        validateTrainArgs(maxSteps, batchSize, logEvery, emaBeta, releaseEverySteps);

        double ema = Double.NaN;
        int step;
        double lastLoss = Double.NaN;

        try {
            for (step = 0; step < maxSteps; step++) {
                lastLoss = trainStep(batchSize);
                ema = updateEma(ema, emaBeta, lastLoss);

                maybeLog(step, logEvery, lastLoss, ema);
                invokeStepHookSafely(stepHook, step, lastLoss, ema);
                maybeReleaseResources(step, releaseEverySteps);

                if (shouldEarlyStop(targetEmaLoss, ema)) {
                    break;
                }
            }
        } finally {
            Tensor.backend().releaseResources();
        }

        int stepsRun = computeStepsRun(step, maxSteps);
        return new TrainingResult(stepsRun, lastLoss, ema);
    }

    private static void validateTrainArgs(int maxSteps, int batchSize, int logEvery, double emaBeta, int releaseEverySteps) {
        if (maxSteps <= 0) throw new IllegalArgumentException("maxSteps must be > 0");
        if (batchSize <= 0) throw new IllegalArgumentException("batchSize must be > 0");
        if (logEvery <= 0) throw new IllegalArgumentException("logEvery must be > 0");
        if (emaBeta <= 0.0 || emaBeta >= 1.0) throw new IllegalArgumentException("emaBeta must be in (0,1)");
        if (releaseEverySteps < 0) throw new IllegalArgumentException("releaseEverySteps must be >= 0");
    }

    private static double updateEma(double ema, double emaBeta, double lastLoss) {
        return Double.isNaN(ema) ? lastLoss : (emaBeta * ema + (1.0 - emaBeta) * lastLoss);
    }

    private static void maybeLog(int step, int logEvery, double lastLoss, double ema) {
        if (step % logEvery == 0) {
            System.out.printf("step=%d loss=%.6f ema=%.6f%n", step, lastLoss, ema);
        }
    }

    private static void invokeStepHookSafely(StepHook stepHook, int step, double lastLoss, double ema) {
        if (stepHook == null) return;
        try {
            stepHook.onStep(step, lastLoss, ema);
        } catch (Exception e) {
            throw new RuntimeException("Step hook failed at step " + step, e);
        }
    }

    private static void maybeReleaseResources(int step, int releaseEverySteps) {
        if (releaseEverySteps > 0 && step > 0 && step % releaseEverySteps == 0) {
            Tensor.backend().releaseResources();
        }
    }

    private static boolean shouldEarlyStop(Double targetEmaLoss, double ema) {
        return targetEmaLoss != null && ema <= targetEmaLoss;
    }

    private static int computeStepsRun(int step, int maxSteps) {
        return (step == maxSteps) ? maxSteps : (step + 1);
    }
}
