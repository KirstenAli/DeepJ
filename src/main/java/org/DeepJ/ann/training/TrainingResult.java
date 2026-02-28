package org.DeepJ.ann.training;

public record TrainingResult(int steps, double lastLoss, double emaLoss) {}
