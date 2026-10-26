package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

public final class ReductionOps {
    private ReductionOps() {}

    public static Tensor rows(Tensor input) {
        int rows = input.matrixRows(), cols = input.matrixCols();
        return UnaryOps.record(input, Tensor.backend().sumRows(input),
                g -> Tensor.ones(rows, cols).multiplyBroadcastRows(g));
    }

    public static Tensor columns(Tensor input) {
        int rows = input.matrixRows(), cols = input.matrixCols();
        return UnaryOps.record(input, Tensor.backend().sumAlongRows(input),
                g -> Tensor.ones(rows, cols).multiplyBroadcastCols(g));
    }

    public static Tensor mean(Tensor input) {
        int rows = input.matrixRows(), cols = input.matrixCols();
        return UnaryOps.record(input, Tensor.backend().meanAlongRows(input),
                g -> Tensor.ones(rows, cols).multiplyBroadcastCols(g).divideScalar(cols));
    }

    public static Tensor variance(Tensor input) {
        return UnaryOps.record(input, Tensor.backend().varianceAlongRows(input),
                g -> input.subtractBroadcastCols(input.meanAlongRows())
                        .multiplyBroadcastCols(g).multiplyScalar(2.0f / input.matrixCols()), input);
    }

    public static Tensor sum(Tensor input) {
        return columns(input).sumRows();
    }
}
