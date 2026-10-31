package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;

/** Independent scalar FP64 diagnostic; never used by production attention. */
final class AttentionDoubleReference {
    private static final int WIDTH = 64;

    record Inputs(Tensor query, Tensor key, Tensor value, Tensor upstream,
                  int heads, int keyValueHeads, float scale) {
        Inputs(Tensor query, Tensor key, Tensor value, Tensor upstream, int heads, float scale) {
            this(query, key, value, upstream, heads, heads, scale);
        }
    }
    record Result(double[] output, double[] query, double[] key, double[] value) {}

    private AttentionDoubleReference() {}

    static Result calculate(Inputs inputs) {
        int count = inputs.query().rawData().length;
        Result result = new Result(new double[count], new double[count],
                new double[inputs.key().rawData().length], new double[inputs.value().rawData().length]);
        int length = inputs.query().matrixRows() / inputs.heads();
        for (int head = 0; head < inputs.heads(); head++) {
            int keyBase = head / (inputs.heads() / inputs.keyValueHeads()) * length;
            accumulateHead(inputs, result, head * length, keyBase, length);
        }
        return result;
    }

    private static void accumulateHead(Inputs inputs, Result result, int base, int keyBase, int length) {
        for (int row = 0; row < length; row++) {
            double[] probabilities = probabilities(inputs, base, keyBase, row);
            double[] upstream = upstream(inputs, base, keyBase, row);
            double delta = dot(probabilities, upstream);
            for (int column = 0; column <= row; column++) {
                double derivative = probabilities[column] * (upstream[column] - delta) * inputs.scale();
                accumulate(inputs, result, base + row, keyBase + column, probabilities[column], derivative);
            }
        }
    }

    private static double[] probabilities(Inputs inputs, int base, int keyBase, int row) {
        double[] scores = new double[row + 1];
        double maximum = Double.NEGATIVE_INFINITY;
        for (int column = 0; column <= row; column++) {
            scores[column] = dot(inputs.query(), base + row, inputs.key(), keyBase + column) * inputs.scale();
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

    private static double[] upstream(Inputs inputs, int base, int keyBase, int row) {
        double[] result = new double[row + 1];
        for (int column = 0; column <= row; column++) {
            result[column] = dot(inputs.upstream(), base + row, inputs.value(), keyBase + column);
        }
        return result;
    }

    private static double dot(Tensor left, int row, Tensor right, int column) {
        double sum = 0;
        for (int index = 0; index < WIDTH; index++) {
            sum += (double) left.rawData()[row * WIDTH + index] * right.rawData()[column * WIDTH + index];
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
            result.output()[query] += probability * inputs.value().rawData()[key];
            result.query()[query] += derivative * inputs.key().rawData()[key];
            result.key()[key] += derivative * inputs.query().rawData()[query];
            result.value()[key] += probability * inputs.upstream().rawData()[query];
        }
    }
}
