package io.github.kirstenali.deepj.tokenizers.bpe;

public record TokenPair(int left, int right) implements Comparable<TokenPair> {

    @Override
    public int compareTo(TokenPair other) {
        int leftCompare = Integer.compare(this.left, other.left);
        if (leftCompare != 0) {
            return leftCompare;
        }
        return Integer.compare(this.right, other.right);
    }
}