package io.github.kirstenali.deepj.transformer.embeddings;

import io.github.kirstenali.deepj.tensor.Tensor;

/** Short-lived inference rows, not a growing table or a trainable parameter. */
record RotaryPositionRows(Tensor cosine, Tensor sine) {
    static RotaryPositionRows at(int headDimension, int length, int offset) {
        var rows = new RotaryPositionRows(Tensor.zeros(length, headDimension / 2),
                Tensor.zeros(length, headDimension / 2));
        for (int row = 0; row < length; row++) {
            rows.populate(row, offset + row, headDimension);
        }
        return rows;
    }

    private void populate(int row, int position, int headDimension) {
        for (int dimension = 0; dimension < headDimension / 2; dimension++) {
            float theta = (float) (position / Math.pow(10_000.0f, (2.0f * dimension) / headDimension));
            int index = row * cosine.matrixCols() + dimension;
            cosine.rawData()[index] = (float) Math.cos(theta);
            sine.rawData()[index] = (float) Math.sin(theta);
        }
    }
}
