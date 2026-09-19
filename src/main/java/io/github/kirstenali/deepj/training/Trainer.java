package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.tensor.Tensor;

public final class Trainer {

    private static final int DEFAULT_RELEASE_EVERY_STEPS = 25;

    @FunctionalInterface
    public interface StepFunction {

        float trainStep(int batchSize);
    }

    @FunctionalInterface
    public interface StepHook {
        void onStep(int step, float loss, float ema) throws Exception;
    }

    private final StepFunction stepFn;

    public Trainer(StepFunction stepFn) {
        if (stepFn == null) throw new IllegalArgumentException("stepFn must not be null");
        this.stepFn = stepFn;
    }

    public float trainStep(int batchSize) {
        return stepFn.trainStep(batchSize);
    }

    public TrainingResult train(
            int maxSteps,
            int batchSize,
            int logEvery,
            float emaBeta,
            Float targetEmaLoss
    ) {
        return train(maxSteps, batchSize, logEvery, emaBeta, targetEmaLoss, DEFAULT_RELEASE_EVERY_STEPS, null);
    }

    public TrainingResult train(
            int maxSteps,
            int batchSize,
            int logEvery,
            float emaBeta,
            Float targetEmaLoss,
            int releaseEverySteps
    ) {
        return train(maxSteps, batchSize, logEvery, emaBeta, targetEmaLoss, releaseEverySteps, null);
    }

    public TrainingResult train(
            int maxSteps,
            int batchSize,
            int logEvery,
            float emaBeta,
            Float targetEmaLoss,
            StepHook stepHook
    ) {
        return train(maxSteps, batchSize, logEvery, emaBeta, targetEmaLoss, DEFAULT_RELEASE_EVERY_STEPS, stepHook);
    }

    public TrainingResult train(
            int maxSteps,
            int batchSize,
            int logEvery,
            float emaBeta,
            Float targetEmaLoss,
            int releaseEverySteps,
            StepHook stepHook
    ) {
        validateTrainArgs(maxSteps, batchSize, logEvery, emaBeta, releaseEverySteps);
        try {
            return runTraining(maxSteps, batchSize, logEvery, emaBeta,
                    targetEmaLoss, releaseEverySteps, stepHook);
        } finally {
            Tensor.backend().releaseResources();
        }
    }

    private TrainingResult runTraining(int maxSteps, int batchSize, int logEvery, float emaBeta,
                                       Float targetEmaLoss, int releaseEverySteps, StepHook stepHook) {
        float ema = Float.NaN;
        float loss = Float.NaN;
        int step;
        for (step = 0; step < maxSteps; step++) {
            loss = trainStep(batchSize);
            ema = updateEma(ema, emaBeta, loss);
            afterStep(stepHook, step, logEvery, releaseEverySteps, loss, ema);
            if (shouldEarlyStop(targetEmaLoss, ema)) break;
        }
        return new TrainingResult(computeStepsRun(step, maxSteps), loss, ema);
    }

    private static void afterStep(StepHook hook, int step, int logEvery,
                                  int releaseEverySteps, float loss, float ema) {
        maybeLog(step, logEvery, loss, ema);
        invokeStepHookSafely(hook, step, loss, ema);
        maybeReleaseResources(step, releaseEverySteps);
    }

    private static void validateTrainArgs(int maxSteps, int batchSize, int logEvery, float emaBeta, int releaseEverySteps) {
        if (maxSteps <= 0) throw new IllegalArgumentException("maxSteps must be > 0");
        if (batchSize <= 0) throw new IllegalArgumentException("batchSize must be > 0");
        if (logEvery <= 0) throw new IllegalArgumentException("logEvery must be > 0");
        if (emaBeta <= 0.0 || emaBeta >= 1.0) throw new IllegalArgumentException("emaBeta must be in (0,1)");
        if (releaseEverySteps < 0) throw new IllegalArgumentException("releaseEverySteps must be >= 0");
    }

    private static float updateEma(float ema, float emaBeta, float lastLoss) {
        return Float.isNaN(ema) ? lastLoss : (emaBeta * ema + (1.0f - emaBeta) * lastLoss);
    }

    private static void maybeLog(int step, int logEvery, float lastLoss, float ema) {
        if (step % logEvery == 0) {
            System.out.printf("step=%d loss=%.6f ema=%.6f%n", step, lastLoss, ema);
        }
    }

    private static void invokeStepHookSafely(StepHook stepHook, int step, float lastLoss, float ema) {
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

    private static boolean shouldEarlyStop(Float targetEmaLoss, float ema) {
        return targetEmaLoss != null && ema <= targetEmaLoss;
    }

    private static int computeStepsRun(int step, int maxSteps) {
        return (step == maxSteps) ? maxSteps : (step + 1);
    }
}
