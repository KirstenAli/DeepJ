package io.github.kirstenali.deepj.nn;

/** Floor-mode spatial window with symmetric zero padding and optional dilation. */
public record Window2D(int height, int width, int strideHeight, int strideWidth,
                       int paddingHeight, int paddingWidth, int dilationHeight, int dilationWidth) {
    public Window2D(int height, int width) { this(height, width, 1, 1, 0, 0, 1, 1); }

    public Window2D {
        NeuralChecks.positive(height, "kernel height");
        NeuralChecks.positive(width, "kernel width");
        NeuralChecks.positive(strideHeight, "height stride");
        NeuralChecks.positive(strideWidth, "width stride");
        NeuralChecks.positive(dilationHeight, "height dilation");
        NeuralChecks.positive(dilationWidth, "width dilation");
        if (paddingHeight < 0 || paddingWidth < 0) throw new IllegalArgumentException("Padding cannot be negative");
        Math.multiplyExact(height, width);
    }

    public int outputHeight(int input) { return output(input, height, strideHeight, paddingHeight, dilationHeight); }
    public int outputWidth(int input) { return output(input, width, strideWidth, paddingWidth, dilationWidth); }
    public int area() { return Math.multiplyExact(height, width); }

    private static int output(int input, int kernel, int stride, int padding, int dilation) {
        NeuralChecks.positive(input, "spatial input size");
        long available = input + 2L * padding - (1L + (kernel - 1L) * dilation);
        if (available < 0) throw new IllegalArgumentException("Effective kernel exceeds padded input");
        return Math.toIntExact(available / stride + 1);
    }
}
