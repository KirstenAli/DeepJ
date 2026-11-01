package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.tensor.Tensor;

/** Gathered [batch, channels, positions, kernel] windows, entirely on the selected backend. */
record SpatialPatches(Tensor values, int height, int width) {
    static SpatialPatches gather(Tensor input, Window2D window, float paddingValue) {
        NeuralChecks.rank(input, 4);
        int height = window.outputHeight(input.size(2)), width = window.outputWidth(input.size(3));
        int spatial = Math.multiplyExact(input.size(2), input.size(3));
        Tensor flat = input.reshape(input.size(0), input.size(1), spatial);
        Tensor padding = Tensor.backend().zeros(input.dtype(), input.size(0), input.size(1), 1);
        if (paddingValue != 0) padding = padding.addScalar(paddingValue);
        Tensor padded = TensorJoin.concatenate(2, flat, padding);
        int[] indices = indices(input, window, height, width);
        Tensor values = padded.indexSelect(2, indices)
                .reshape(input.size(0), input.size(1), Math.multiplyExact(height, width), window.area());
        return new SpatialPatches(values, height, width);
    }

    private static int[] indices(Tensor input, Window2D window, int height, int width) {
        int[] indices = new int[Math.multiplyExact(Math.multiplyExact(height, width), window.area())];
        for (int index = 0; index < indices.length; index++) {
            int position = index / window.area(), kernel = index % window.area();
            long y = (long) (position / width) * window.strideHeight() - window.paddingHeight()
                    + (long) (kernel / window.width()) * window.dilationHeight();
            long x = (long) (position % width) * window.strideWidth() - window.paddingWidth()
                    + (long) (kernel % window.width()) * window.dilationWidth();
            indices[index] = address(input, y, x);
        }
        return indices;
    }

    private static int address(Tensor input, long y, long x) {
        if (y < 0 || y >= input.size(2) || x < 0 || x >= input.size(3)) {
            return Math.multiplyExact(input.size(2), input.size(3));
        }
        return Math.toIntExact(y * input.size(3) + x);
    }
}
