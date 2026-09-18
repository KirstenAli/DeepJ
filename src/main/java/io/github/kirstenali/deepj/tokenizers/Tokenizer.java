package io.github.kirstenali.deepj.tokenizers;

public interface Tokenizer {
    int[] encode(String text);
    String decode(int[] ids);
    int vocabSize();

    /** Returns true when generation should stop before appending this token. */
    default boolean isEndOfSequence(int tokenId) {
        return false;
    }
}
