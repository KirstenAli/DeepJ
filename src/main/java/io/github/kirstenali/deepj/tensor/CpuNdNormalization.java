package io.github.kirstenali.deepj.tensor;

/** Last-axis CPU normalization, with FLOAT32 statistics and no intermediate compact rounding. */
public final class CpuNdNormalization {
    private CpuNdNormalization() {}

    public static NormalizationResult apply(Tensor input, Tensor gamma, Tensor beta, float epsilon, boolean centered) {
        NormalizationChecks.forward(input, gamma, beta, epsilon, centered);
        materialize(input, gamma, beta);
        DType type = NormalizationChecks.outputType(input, gamma, beta, centered);
        Tensor output = new Tensor(type, input.shape()), normalized = new Tensor(DType.FLOAT32, input.shape());
        Tensor denominator = new Tensor(DType.FLOAT32, NormalizationChecks.statisticsShape(input));
        var result = new NormalizationResult(output, normalized, denominator);
        if (!CpuNativeNormalization.forward(input, gamma, beta, result, epsilon, centered)) {
            normalizeRows(input, gamma, beta, result, epsilon, centered);
        }
        return result;
    }

    private static void normalizeRows(Tensor input, Tensor gamma, Tensor beta, NormalizationResult result,
                                       float epsilon, boolean centered) {
        int width = input.size(-1), rows = input.numel() / width;
        Tensor gain = gain(gamma, rows);
        if (input.numel() >= 16384 && rows > 1) {
            CpuRanges.balance(rows, row -> normalizeRow(input, gain, beta, result, row, width, epsilon, centered));
            return;
        }
        for (int row = 0; row < rows; row++) {
            normalizeRow(input, gain, beta, result, row, width, epsilon, centered);
        }
    }

    private static Tensor gain(Tensor gamma, int rows) {
        return rows > 1 && gamma.dtype() != DType.FLOAT32
                ? Tensor.from(gamma.toFloatArray(), gamma.shape()) : gamma;
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
        if (CpuNativeNormalization.backward(gradient, normalized, denominator, gamma, output, centered)) return output;
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
        if (tensor.dtype() == DType.FLOAT32 && tensor.isContiguous()) {
            return tensor.memory().floats()[tensor.storageOffset() + index];
        }
        if (tensor.isContiguous()) return tensor.memory().get(tensor.storageOffset() + index);
        return tensor.memory().get(tensor.layout().addressAt(index));
    }

    private static void materialize(Tensor... tensors) {
        for (Tensor tensor : tensors) {
            if (tensor != null) tensor.materialize();
        }
    }
}
