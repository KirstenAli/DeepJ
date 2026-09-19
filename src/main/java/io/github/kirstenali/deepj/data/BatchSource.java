package io.github.kirstenali.deepj.data;

@FunctionalInterface
public interface BatchSource {

    Batch nextBatch(int batchSize);
}
