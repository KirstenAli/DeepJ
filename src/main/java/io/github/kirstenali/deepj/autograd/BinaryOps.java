package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.function.Function;

public final class BinaryOps {
    private BinaryOps() {}

    public static Tensor matmul(Tensor a, Tensor b) {
        return record(Tensor.backend().matmul(a, b), a, b,
                g -> Tensor.backend().batchedMatmul(g, b, 1, false, true),
                g -> Tensor.backend().batchedMatmul(a, g, 1, true, false));
    }

    public static Tensor add(Tensor a, Tensor b) {
        return record(Tensor.backend().add(a, b), a, b, g -> g, g -> g);
    }

    public static Tensor subtract(Tensor a, Tensor b) {
        return record(Tensor.backend().subtract(a, b), a, b, g -> g, Tensor::neg);
    }

    public static Tensor multiply(Tensor a, Tensor b) {
        return record(Tensor.backend().multiply(a, b), a, b, g -> g.multiply(b), g -> g.multiply(a));
    }

    public static Tensor divide(Tensor a, Tensor b) {
        return record(Tensor.backend().divide(a, b), a, b,
                g -> g.divide(b), g -> g.multiply(a).divide(b.multiply(b)).neg());
    }

    static Tensor record(Tensor output, Tensor a, Tensor b,
                          Function<Tensor, Tensor> left, Function<Tensor, Tensor> right) {
        return Autograd.record(output, g -> new Tensor[]{
                a.requiresGrad() ? left.apply(g) : null,
                b.requiresGrad() ? right.apply(g) : null}, a, b);
    }
}
