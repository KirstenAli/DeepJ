package io.github.kirstenali.deepj.tensor;

/** Model activations use[batch,time,width]; historical single-sequence matrices remain explicit. */
public final class SequenceLayout {
    private SequenceLayout() {}

    public static int length(Tensor input) {
        requireSequence(input);
        return input.size(-2);
    }

    public static int batches(Tensor input) {
        requireSequence(input);
        return input.rank() == 2 ? 1 : input.size(0);
    }

    public static void requireWidth(Tensor input, int width) {
        requireSequence(input);
        if (input.size(-1) != width) throw new IllegalArgumentException("Expected final sequence width=" + width);
    }

    private static void requireSequence(Tensor input) {
        if (input.rank() != 2 && input.rank() != 3) {
            throw new IllegalArgumentException("Expected[time,width]or[batch,time,width]activations");
        }
    }
}
