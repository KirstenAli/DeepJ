package io.github.kirstenali.deepj.training;

/** Aggregate causal language-model evaluation metrics. */
public record EvaluationResult(double loss, double perplexity, long tokens) {}
