package io.github.kirstenali.deepj.training;

public record TrainingResult(int steps, float lastLoss, float emaLoss) {}
