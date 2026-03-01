package io.github.kirstenali.deepj.tokenizers;

public interface Tokenizer {
    int[] encode(String text);

    String decode(int[] ids);
}
