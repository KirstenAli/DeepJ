package io.github.kirstenali.deepj.examples;

/** Evaluates the looped decoder on the unchanged held-out FineWeb-Edu split. */
public final class EvaluateDeepJLoop100M {

    private EvaluateDeepJLoop100M() {}

    public static void main(String[] args) throws Exception {
        EvaluateDeepJ100M.run("checkpoints/deepj-loop-100m/pretrain");
    }
}
