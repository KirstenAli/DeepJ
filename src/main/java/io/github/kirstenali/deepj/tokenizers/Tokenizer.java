package io.github.kirstenali.deepj.tokenizers;

public interface Tokenizer {
    int[] encode(String text);
    String decode(int[] ids);
    int vocabSize();

    /** Decodes a token prefix for display while generation is still in progress. */
    default String decodePartial(int[] ids) {
        return decode(ids);
    }

    default boolean isEndOfSequence(int tokenId) {
        return false;
    }
}
