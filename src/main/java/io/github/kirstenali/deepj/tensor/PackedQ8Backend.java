package io.github.kirstenali.deepj.tensor;

/** Backend capability for packed Q8_0 inference operands. */
public interface PackedQ8Backend {
    Tensor projectQ8(Tensor input, Tensor packed, int rows, int width);
    Tensor gatherQ8(Tensor packed, int rows, int width, int[] tokens);
}
