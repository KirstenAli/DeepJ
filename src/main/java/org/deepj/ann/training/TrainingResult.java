package org.deepj.ann.training;

public record TrainingResult(int steps, double lastLoss, double emaLoss) {}
