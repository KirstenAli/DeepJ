package io.github.kirstenali.deepj.nn.transformer.qwen3;

import java.util.Arrays;

final class ReferenceAttention {
    private ReferenceAttention() {}

    static double[][] normalized(double[][] projected, double[] gain, double theta) {
        double[][] result = new double[projected.length][projected[0].length];
        for (int time = 0; time < result.length; time++) {
            for (int offset = 0; offset < result[time].length; offset += gain.length) {
                double[] head = ReferenceMath.normalize(Arrays.copyOfRange(projected[time], offset,
                        offset + gain.length), gain);
                rotate(head, time, theta);
                System.arraycopy(head, 0, result[time], offset, head.length);
            }
        }
        return result;
    }

    private static void rotate(double[] head, int position, double theta) {
        int half = head.length / 2;
        for (int index = 0; index < half; index++) {
            double angle = position / Math.pow(theta, 2.0 * index / head.length);
            double first = head[index], second = head[index + half];
            head[index] = first * Math.cos(angle) - second * Math.sin(angle);
            head[index + half] = first * Math.sin(angle) + second * Math.cos(angle);
        }
    }

    static double[][] apply(Qwen3Config config, double[][] query, double[][] key, double[][] value, int window) {
        double[][] output = new double[query.length][config.nHeads() * config.headDim()];
        for (int time = 0; time < query.length; time++) {
            for (int head = 0; head < config.nHeads(); head++) {
                int keyHead = head / (config.nHeads() / config.kvHeads());
                accumulate(config, query[time], key, value, output[time], time, head, keyHead, window);
            }
        }
        return output;
    }

    private static void accumulate(Qwen3Config config, double[] query, double[][] key, double[][] value,
                                   double[] result, int time, int head, int keyHead, int window) {
        int start = Math.max(0, time - window + 1), width = config.headDim();
        double[] scores = new double[time - start + 1];
        for (int index = 0; index < scores.length; index++) {
            scores[index] = dot(query, key[start + index], head * width, keyHead * width, width) / Math.sqrt(width);
        }
        double max = Arrays.stream(scores).max().orElseThrow(), sum = 0;
        for (double score : scores) { sum += Math.exp(score - max); }
        for (int index = 0; index < scores.length; index++) {
            double probability = Math.exp(scores[index] - max) / sum;
            for (int dimension = 0; dimension < width; dimension++) {
                result[head * width + dimension] += probability * value[start + index][keyHead * width + dimension];
            }
        }
    }

    private static double dot(double[] query, double[] key, int queryOffset, int keyOffset, int width) {
        double sum = 0;
        for (int dimension = 0; dimension < width; dimension++) {
            sum += query[queryOffset + dimension] * key[keyOffset + dimension];
        }
        return sum;
    }
}
