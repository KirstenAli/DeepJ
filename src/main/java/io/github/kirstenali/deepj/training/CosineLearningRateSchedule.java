package io.github.kirstenali.deepj.training;

public record CosineLearningRateSchedule(
        float peakLearningRate,
        float minimumLearningRate,
        int warmupSteps,
        int totalSteps
) {

    public CosineLearningRateSchedule {
        validateRates(peakLearningRate, minimumLearningRate);
        if (totalSteps <= 0) throw new IllegalArgumentException("totalSteps must be > 0");
        if (warmupSteps < 0 || warmupSteps >= totalSteps) {
            throw new IllegalArgumentException("warmupSteps must be in [0, totalSteps)");
        }
    }

    public float learningRate(int step) {
        if (step < 0) throw new IllegalArgumentException("step must be >= 0");
        if (warmupSteps > 0 && step < warmupSteps) {
            return peakLearningRate * (step + 1.0f) / warmupSteps;
        }
        if (step >= totalSteps) return minimumLearningRate;
        double progress = (double) (step - warmupSteps) / (totalSteps - warmupSteps);
        double cosine = 0.5 * (1.0 + Math.cos(Math.PI * progress));
        return (float) (minimumLearningRate
                + (peakLearningRate - minimumLearningRate) * cosine);
    }

    private static void validateRates(float peak, float minimum) {
        if (!Float.isFinite(peak) || peak <= 0.0f) {
            throw new IllegalArgumentException("peakLearningRate must be finite and > 0");
        }
        if (!Float.isFinite(minimum) || minimum < 0.0f || minimum > peak) {
            throw new IllegalArgumentException("minimumLearningRate must be finite and in [0, peak]");
        }
    }
}
