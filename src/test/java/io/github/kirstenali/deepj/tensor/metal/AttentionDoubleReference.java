package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;

/** Independent scalar FP64 diagnostic; never used by production attention. */
final class AttentionDoubleReference {
    private static final int WIDTH = 64;

    record Inputs(Tensor query, Tensor key, Tensor value, Tensor upstream, int heads, float scale) {}
    record Result(double[] output, double[] query, double[] key, double[] value) {}

    private AttentionDoubleReference() {}

    static Result calculate(Inputs inputs) {
        int count = inputs.query().data.length;
        Result result = new Result(new double[count], new double[count], new double[count], new double[count]);
        int length = inputs.query().rows / inputs.heads();
        for (int head = 0; head < inputs.heads(); head++) {
            accumulateHead(inputs, result, head * length, length);
        }
        return result;
    }

    private static void accumulateHead(Inputs inputs, Result result, int base, int length) {
        for (int row = 0; row < length; row++) {
            double[] probabilities = probabilities(inputs, base, row);
            double[] upstream = upstream(inputs, base, row);
            double delta = dot(probabilities, upstream);
            for (int column = 0; column <= row; column++) {
                double derivative = probabilities[column] * (upstream[column] - delta) * inputs.scale();
                accumulate(inputs, result, base + row, base + column, probabilities[column], derivative);
            }
        }
    }

    private static double[] probabilities(Inputs inputs, int base, int row) {
        double[] scores = new double[row + 1];
        double maximum = Double.NEGATIVE_INFINITY;
        for (int column = 0; column <= row; column++) {
            scores[column] = dot(inputs.query(), base + row, inputs.key(), base + column) * inputs.scale();
            maximum = Math.max(maximum, scores[column]);
        }
        double sum = exponentiate(scores, maximum);
        for (int index = 0; index < scores.length; index++) {
            scores[index] /= sum;
        }
        return scores;
    }

    private static double exponentiate(double[] scores, double maximum) {
        double sum = 0;
        for (int index = 0; index < scores.length; index++) {
            scores[index] = StrictMath.exp(scores[index] - maximum);
            sum += scores[index];
        }
        return sum;
    }

    private static double[] upstream(Inputs inputs, int base, int row) {
        double[] result = new double[row + 1];
        for (int column = 0; column <= row; column++) {
            result[column] = dot(inputs.upstream(), base + row, inputs.value(), base + column);
        }
        return result;
    }

    private static double dot(Tensor left, int row, Tensor right, int column) {
        double sum = 0;
        for (int index = 0; index < WIDTH; index++) {
            sum += (double) left.data[row * WIDTH + index] * right.data[column * WIDTH + index];
        }
        return sum;
    }

    private static double dot(double[] left, double[] right) {
        double sum = 0;
        for (int index = 0; index < left.length; index++) {
            sum += left[index] * right[index];
        }
        return sum;
    }

    private static void accumulate(Inputs inputs, Result result, int row, int column,
                                   double probability, double derivative) {
        for (int index = 0; index < WIDTH; index++) {
            int query = row * WIDTH + index;
            int key = column * WIDTH + index;
            result.output()[query] += probability * inputs.value().data[key];
            result.query()[query] += derivative * inputs.key().data[key];
            result.key()[key] += derivative * inputs.query().data[query];
            result.value()[key] += probability * inputs.upstream().data[query];
        }
    }
}
