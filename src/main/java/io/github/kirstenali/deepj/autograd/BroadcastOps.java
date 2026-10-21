package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

public final class BroadcastOps {
    private BroadcastOps() {}

    public static Tensor add(Tensor a, Tensor b, boolean columns) {
        Tensor output = columns ? Tensor.backend().addBroadcastCols(a, b) : Tensor.backend().addRowVector(a, b);
        return BinaryOps.record(output, a, b, g -> g, g -> reduce(g, columns));
    }

    public static Tensor subtract(Tensor a, Tensor b) {
        return BinaryOps.record(Tensor.backend().subtractBroadcastCols(a, b), a, b,
                g -> g, g -> g.sumAlongRows().neg());
    }

    public static Tensor multiply(Tensor a, Tensor b, boolean columns) {
        return BinaryOps.record(product(a, b, columns), a, b,
                g -> product(g, b, columns), g -> reduce(g.multiply(a), columns));
    }

    public static Tensor divide(Tensor a, Tensor b) {
        return BinaryOps.record(Tensor.backend().divideBroadcastCols(a, b), a, b,
                g -> g.divideBroadcastCols(b),
                g -> g.multiply(a).sumAlongRows().divide(b.multiply(b)).neg());
    }

    private static Tensor product(Tensor a, Tensor b, boolean columns) {
        return columns ? Tensor.backend().multiplyBroadcastCols(a, b) : Tensor.backend().multiplyBroadcastRows(a, b);
    }

    private static Tensor reduce(Tensor gradient, boolean columns) {
        return columns ? gradient.sumAlongRows() : gradient.sumRows();
    }
}
