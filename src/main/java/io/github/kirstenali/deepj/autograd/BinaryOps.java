package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.function.Function;

public final class BinaryOps {
    private BinaryOps() {}

    public static Tensor matmul(Tensor a, Tensor b) {
        if (a.rank() != 2 || b.rank() != 2) return NdMatmulOps.apply(a, b);
        return record(Tensor.backend().matmul(a, b), a, b,
                g -> Tensor.backend().batchedMatmul(g, b, 1, false, true),
                g -> Tensor.backend().batchedMatmul(a, g, 1, true, false));
    }

    public static Tensor add(Tensor a, Tensor b) {
        return recordSaved(Tensor.backend().add(a, b), a, b, g -> g, g -> g);
    }

    public static Tensor subtract(Tensor a, Tensor b) {
        return recordSaved(Tensor.backend().subtract(a, b), a, b, g -> g, Tensor::neg);
    }

    public static Tensor multiply(Tensor a, Tensor b) {
        return record(Tensor.backend().multiply(a, b), a, b, g -> g.multiply(b), g -> g.multiply(a));
    }

    public static Tensor divide(Tensor a, Tensor b) {
        return record(Tensor.backend().divide(a, b), a, b,
                g -> g.divide(b), g -> g.multiply(a).divide(GradientDType.square(b)).neg());
    }

    static Tensor record(Tensor output, Tensor a, Tensor b,
                          Function<Tensor, Tensor> left, Function<Tensor, Tensor> right) {
        return recordSaved(output, a, b, left, right, a, b);
    }

    static Tensor recordSaved(Tensor output, Tensor a, Tensor b,
                              Function<Tensor, Tensor> left, Function<Tensor, Tensor> right, Tensor... saved) {
        if (!Autograd.tracks(a, b)) return output;
        Function<Tensor, Tensor> leftRule = a.requiresGrad() ? left : null;
        Function<Tensor, Tensor> rightRule = b.requiresGrad() ? right : null;
        int[] leftShape = output.hasSameShape(a) ? null : a.shape();
        int[] rightShape = output.hasSameShape(b) ? null : b.shape();
        return Autograd.recordBuiltinSaved(output, g -> new Tensor[]{
                derivative(leftRule, g, leftShape), derivative(rightRule, g, rightShape)},
                new Tensor[]{a, b}, saved);
    }

    private static Tensor derivative(Function<Tensor, Tensor> rule, Tensor gradient, int[] shape) {
        if (rule == null) return null;
        Tensor result = rule.apply(gradient);
        return shape == null ? result : Tensor.backend().sumToShape(result, shape);
    }
}
