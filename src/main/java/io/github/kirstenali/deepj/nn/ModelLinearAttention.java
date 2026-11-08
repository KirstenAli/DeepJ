package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Combines the inference denominator with the value product, sharing the same key reduction. */
final class ModelLinearAttention {
    private ModelLinearAttention() {}

    static Tensor apply(Tensor query, Tensor key, Tensor value, float epsilon) {
        Tensor transposed = value.to(DType.FLOAT32).transpose(-1, -2);
        int[] shape = transposed.shape();
        shape[shape.length - 2] = 1;
        Tensor ones = Tensor.backend().expand(Tensor.from(new float[]{1}, DType.FLOAT32, 1), shape);
        Tensor scores = ModelOps.cat(-2, transposed, ones).matmul(key).matmul(query.transpose(-1, -2));
        int width = value.size(-1);
        Tensor denominator = scores.slice(-2, width, width + 1).addScalar(epsilon);
        return scores.slice(-2, 0, width).divide(denominator).transpose(-1, -2).to(value.dtype());
    }
}
