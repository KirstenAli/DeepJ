package io.github.kirstenali.deepj.data;

public interface StatefulBatchSource extends BatchSource {

    long randomState();

    void restoreRandomState(long state);
}
