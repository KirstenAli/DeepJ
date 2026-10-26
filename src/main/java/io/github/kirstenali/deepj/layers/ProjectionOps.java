package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.autograd.LayoutOps;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Independent leading positions share one projection; public activation axes remain intact. */
public final class ProjectionOps {
    private ProjectionOps() {}

    public static Tensor linear(Tensor input, Tensor weight) { return apply(input, weight, null, false); }
    public static Tensor affine(Tensor input, Tensor weight, Tensor bias) { return apply(input, weight, bias, false); }
    public static Tensor tied(Tensor input, Tensor embedding) { return apply(input, embedding, null, true); }

    private static Tensor apply(Tensor input, Tensor weight, Tensor bias, boolean transpose) {
        int width = transpose ? weight.matrixCols() : weight.matrixRows();
        Tensor matrix = positions(input, width);
        Tensor result = transpose ? LayoutOps.batchedMatmul(matrix, weight, 1, false, true) : matrix.matmul(weight);
        return restore(input, bias == null ? result : addBias(result, bias));
    }

    private static Tensor positions(Tensor input, int width) {
        if (input.rank() == 0 || input.size(-1) != width) {
            throw new IllegalArgumentException("Projection input must have final width=" + width);
        }
        return input.rank() == 2 ? input : input.reshape(input.numel() / width, width);
    }

    private static Tensor restore(Tensor input, Tensor result) {
        if (input.rank() == 2) return result;
        int[] shape = input.shape();
        shape[shape.length - 1] = result.size(-1);
        return result.reshape(shape);
    }

    private static Tensor addBias(Tensor result, Tensor bias) {
        if (bias.rank() != 2 || bias.size(0) != 1 || bias.size(1) != result.size(-1)) {
            throw new IllegalArgumentException("Projection bias must have shape[1,outputWidth]");
        }
        return result.dtype() == DType.FLOAT32 && bias.dtype() == DType.FLOAT32
                ? result.addRowVector(bias) : result.add(bias.reshape(bias.size(-1)));
    }
}
