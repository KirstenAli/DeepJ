package io.github.kirstenali.deepj.examples;

enum DeepJ100MStage {
    PRETRAIN("pretrain", "fineweb-edu.txt", "deepj-90m/pretrain", 1_757_813, 1e-4f, 2_000, 100L),
    MIDTRAIN("midtrain", "midtrain-train.txt", "deepj-100m/pretrain", 256_000, 5e-5f, 500, 101L),
    FINE_TUNE("sft", "sft-train.txt", "deepj-100m/midtrain", 22_500, 2e-5f, 500, 102L);

    final String directory;
    final String corpus;
    final String base;
    final int steps;
    final float learningRate;
    final int warmup;
    final long seed;

    DeepJ100MStage(String directory, String corpus, String base,
                   int steps, float learningRate, int warmup, long seed) {
        this.directory = directory;
        this.corpus = corpus;
        this.base = base;
        this.steps = steps;
        this.learningRate = learningRate;
        this.warmup = warmup;
        this.seed = seed;
    }
}
