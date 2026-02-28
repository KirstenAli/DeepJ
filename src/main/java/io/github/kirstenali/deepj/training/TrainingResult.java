package io.github.kirstenali.deepj.training;

public record TrainingResult(int steps, double lastLoss, double emaLoss) {}
