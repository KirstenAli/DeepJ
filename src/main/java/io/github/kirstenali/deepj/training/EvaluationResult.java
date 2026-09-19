package io.github.kirstenali.deepj.training;

public record EvaluationResult(double loss, double perplexity, long tokens) {}
