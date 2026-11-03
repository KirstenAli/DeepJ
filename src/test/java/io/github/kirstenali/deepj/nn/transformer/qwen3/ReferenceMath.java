package io.github.kirstenali.deepj.nn.transformer.qwen3;

/** Independent double-precision reference arithmetic; never calls DeepJ tensor operators. */
final class ReferenceMath {
    private ReferenceMath() {}

    static double[][] linear(double[][] input, double[] weight, int outputs) {
        double[][] result = new double[input.length][outputs];
        for (int row = 0; row < input.length; row++) { product(input[row], weight, result[row]); }
        return result;
    }

    private static void product(double[] input, double[] weight, double[] output) {
        for (int column = 0; column < output.length; column++) {
            for (int inner = 0; inner < input.length; inner++) {
                output[column] += input[inner] * weight[inner * output.length + column];
            }
        }
    }

    static double[][] norm(double[][] input, double[] gain) {
        double[][] result = new double[input.length][];
        for (int row = 0; row < input.length; row++) { result[row] = normalize(input[row], gain); }
        return result;
    }

    static double[] normalize(double[] input, double[] gain) {
        double square = 0;
        for (double value : input) { square += value * value; }
        double inverse = 1 / Math.sqrt(square / input.length + 1e-6);
        double[] result = new double[input.length];
        for (int index = 0; index < result.length; index++) { result[index] = input[index] * inverse * gain[index]; }
        return result;
    }

    static double[][] add(double[][] first, double[][] second) {
        double[][] result = new double[first.length][first[0].length];
        for (int row = 0; row < result.length; row++) {
            for (int column = 0; column < result[row].length; column++) {
                result[row][column] = first[row][column] + second[row][column];
            }
        }
        return result;
    }

    static double[][] gated(double[][] gate, double[][] up) {
        double[][] result = new double[gate.length][gate[0].length];
        for (int row = 0; row < result.length; row++) {
            for (int column = 0; column < result[row].length; column++) {
                double value = gate[row][column];
                result[row][column] = value / (1 + Math.exp(-value)) * up[row][column];
            }
        }
        return result;
    }

    static double[] doubles(float[] source) {
        double[] result = new double[source.length];
        for (int index = 0; index < result.length; index++) { result[index] = source[index]; }
        return result;
    }

    static float[] floats(double[][] source) {
        float[] result = new float[source.length * source[0].length];
        for (int row = 0; row < source.length; row++) {
            for (int column = 0; column < source[row].length; column++) {
                result[row * source[row].length + column] = (float) source[row][column];
            }
        }
        return result;
    }
}
