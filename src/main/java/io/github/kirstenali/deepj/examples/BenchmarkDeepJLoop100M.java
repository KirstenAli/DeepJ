package io.github.kirstenali.deepj.examples;

/** Synthetic full training updates; reads no dataset or checkpoint and saves nothing. */
public final class BenchmarkDeepJLoop100M {

    private BenchmarkDeepJLoop100M() {}

    public static void main(String[] args) {
        BenchmarkDeepJ100M.run(DeepJLoop100MConfig.fromSystemProperties(DeepJ100MStage.PRETRAIN));
    }
}
