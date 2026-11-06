package io.github.kirstenali.deepj.tensor;

record CpuProductTile(CpuProductPlan plan, int rows, int columns, int left, int right, int output) {
    int leftRow(int row) { return left + row * plan.leftRowStride; }
    int outputRow(int row) { return output + row * plan.shape.columns(); }

    void store(int row, float[] values) {
        int first = outputRow(row);
        for (int column = 0; column < values.length; column++) {
            plan.output.set(first + column, values[column]);
        }
    }
}
