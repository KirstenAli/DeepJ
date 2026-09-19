package io.github.kirstenali.deepj.training;

public record TrainingProgress(int completedSteps, float lastLoss, float emaLoss) {

    public TrainingProgress {
        if (completedSteps < 0) throw new IllegalArgumentException("completedSteps must be non-negative");
        boolean finite = Float.isFinite(lastLoss) && Float.isFinite(emaLoss);
        boolean unknown = Float.isNaN(lastLoss) && Float.isNaN(emaLoss);
        if (!finite && !unknown) {
            throw new IllegalArgumentException("loss values must both be finite or unknown");
        }
    }

    public static TrainingProgress initial() {
        return new TrainingProgress(0, Float.NaN, Float.NaN);
    }
}
