package io.github.kirstenali.deepj.tensor;

/** General strided/broadcast CPU matmul; the existing FLOAT32 matrix fast path remains separate. */
public final class CpuNdMatmul {
    private CpuNdMatmul() {}

    public static Tensor apply(Tensor left, Tensor right) {
        var shape = new NdMatmulShape(left, right);
        Tensor output = new Tensor(DType.promote(left.dtype(), right.dtype()), shape.output().dimensions());
        left.materialize();
        right.materialize();
        for (int index = 0; index < output.numel(); index++) {
            output.memory().set(index, product(left, right, shape, index));
        }
        return output;
    }

    private static float product(Tensor left, Tensor right, NdMatmulShape shape, int index) {
        int batch = index / (shape.rows() * shape.columns());
        int row = index / shape.columns() % shape.rows(), column = index % shape.columns();
        int aBase = (batch * shape.rows() + row) * shape.inner();
        int bBase = batch * shape.inner() * shape.columns() + column;
        float sum = 0;
        for (int inner = 0; inner < shape.inner(); inner++) {
            float a = left.memory().get(shape.left().addressAt(aBase + inner));
            float b = right.memory().get(shape.right().addressAt(bBase + inner * shape.columns()));
            sum += a * b;
        }
        return sum;
    }
}
