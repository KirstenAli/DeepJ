package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

final class GatherBackward implements LeafGradientRule {
    private final GradientState input;
    private final int rows;
    private final int columns;
    private final int[] indices;

    GatherBackward(Tensor input, int[] indices) {
        this.input = input.gradientState();
        rows = input.matrixRows();
        columns = input.matrixCols();
        this.indices = indices;
    }

    @Override
    public boolean accumulate(Tensor gradient) {
        if (input.node() != null || input.gradient() == gradient) return false;
        input.accumulateRows(rows, columns, indices, gradient);
        return true;
    }

    @Override
    public Tensor[] apply(Tensor gradient) {
        Tensor result = Tensor.zeros(rows, columns);
        Tensor.backend().scatterAddRows(result, indices, gradient);
        return new Tensor[]{result};
    }
}
