package io.github.kirstenali.deepj.tensor;

/** Last-axis CPU normalization, with FLOAT32 statistics and no intermediate compact rounding. */
public final class CpuNdNormalization {
    private CpuNdNormalization() {}

    public static NormalizationResult apply(Tensor input, Tensor gamma, Tensor beta, float epsilon, boolean centered) {
        NormalizationChecks.forward(input, gamma, beta, epsilon, centered);
        materialize(input, gamma, beta);
        DType type = centered ? DType.promote(DType.promote(input.dtype(), gamma.dtype()), beta.dtype())
                : DType.promote(input.dtype(), gamma.dtype());
        Tensor output = new Tensor(type, input.shape()), normalized = new Tensor(DType.FLOAT32, input.shape());
        Tensor denominator = new Tensor(DType.FLOAT32, NormalizationChecks.statisticsShape(input));
        var result = new NormalizationResult(output, normalized, denominator);
        int width = input.size(-1);
        for (int row = 0; row < input.numel() / width; row++) {
            normalizeRow(input, gamma, beta, result, row, width, epsilon, centered);
        }
        return result;
    }

    private static void normalizeRow(Tensor input, Tensor gamma, Tensor beta, NormalizationResult result,
                                        int row, int width, float epsilon, boolean centered) {
        int base = row * width;
        float inverse = 1.0f / width, mean = centered ? mean(input, base, width, inverse) : 0;
        float squares = 0;
        for (int column = 0; column < width; column++) {
            float difference = value(input, base + column) - mean;
            squares += difference * difference;
        }
        float denominator = (float) Math.sqrt(squares * inverse + epsilon);
        result.denominator().memory().set(row, denominator);
        for (int column = 0; column < width; column++) {
            float normalized = (value(input, base + column) - mean) / denominator;
            result.normalized().memory().set(base + column, normalized);
            float output = normalized * value(gamma, column);
            result.output().memory().set(base + column, centered ? output + value(beta, column) : output);
        }
    }

    private static float mean(Tensor input, int base, int width, float inverse) {
        float result = 0;
        for (int column = 0; column < width; column++) {
            result += value(input, base + column);
        }
        return result * inverse;
    }

    public static Tensor backward(Tensor gradient, Tensor normalized, Tensor denominator,
                                    Tensor gamma, boolean centered) {
        NormalizationChecks.backward(gradient, normalized, denominator, gamma);
        materialize(gradient, normalized, denominator, gamma);
        Tensor output = new Tensor(DType.FLOAT32, gradient.shape());
        int width = normalized.size(-1);
        for (int row = 0; row < output.numel() / width; row++) {
            gradientRow(output, gradient, normalized, denominator, gamma, row, width, centered);
        }
        return output;
    }

    private static void gradientRow(Tensor output, Tensor gradient, Tensor normalized, Tensor denominator,
                                    Tensor gamma, int row, int width, boolean centered) {
        int base = row * width;
        float sum = 0, dot = 0, inverse = 1.0f / value(denominator, row);
        for (int column = 0; column < width; column++) {
            float scaled = value(gradient, base + column) * value(gamma, column);
            sum += scaled;
            dot += scaled * value(normalized, base + column);
        }
        for (int column = 0; column < width; column++) {
            float scaled = value(gradient, base + column) * value(gamma, column);
            float norm = value(normalized, base + column);
            float derivative = centered ? inverse * (scaled - sum / width - norm * (dot / width))
                    : (scaled - norm * (dot * (1.0f / width))) / value(denominator, row);
            output.memory().set(base + column, derivative);
        }
    }

    private static float value(Tensor tensor, int index) {
        return tensor.memory().get(tensor.layout().addressAt(index));
    }

    private static void materialize(Tensor... tensors) {
        for (Tensor tensor : tensors) {
            if (tensor != null) tensor.materialize();
        }
    }
}
