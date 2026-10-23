package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;

import java.util.Arrays;
import java.util.Random;

/** Isolates legacy transposes versus transpose flags at actual projection shapes. */
public final class MatmulBackwardCompare {
    private MatmulBackwardCompare() {}

    public static void main(String[] args) {
        if (!MetalBackend.isAvailable()) throw new IllegalStateException("Metal required");
        Tensor.setBackend(new MetalBackend());
        try {
            for (int[] shape : shapes(args[0])) {
                compare(shape[0], shape[1], false);
                compare(shape[0], shape[1], true);
            }
        } finally {
            Tensor.backend().releaseResources();
        }
    }

    private static int[][] shapes(String profile) {
        return switch (profile) {
            case "prism" -> new int[][]{{768, 16_384}, {768, 2_112}, {2_112, 768},
                    {768, 384}, {384, 768}, {768, 192}, {192, 768}};
            case "nova" -> new int[][]{{576, 16_384}, {576, 1_664}, {1_664, 576}, {576, 576}, {576, 192}};
            default -> throw new IllegalArgumentException("Unknown profile: " + profile);
        };
    }

    private static void compare(int width, int output, boolean weightGradient) {
        Random random = new Random(31);
        Tensor a = Tensor.random(1_024, weightGradient ? width : output, random).retainDeviceBuffer();
        Tensor b = Tensor.random(weightGradient ? 1_024 : width, output, random).retainDeviceBuffer();
        double[][] times = samples(a, b, weightGradient);
        System.out.printf("MATMUL width=%d output=%d weightGradient=%s explicitSeconds=%.6f flaggedSeconds=%.6f%n",
                width, output, weightGradient, median(times[0]), median(times[1]));
        Tensor.backend().releaseResources();
    }

    private static double[][] samples(Tensor a, Tensor b, boolean weightGradient) {
        double[][] times = new double[2][20];
        for (int round = 0; round < 28; round++) {
            for (int order = 0; order < 2; order++) {
                int path = (round + order) % 2;
                double seconds = measure(a, b, weightGradient, path == 1);
                if (round >= 8) times[path][round - 8] = seconds;
            }
        }
        return times;
    }

    private static double measure(Tensor a, Tensor b, boolean weightGradient, boolean flagged) {
        long started = System.nanoTime();
        float sum = Tensor.backend().sum(product(a, b, weightGradient, flagged));
        Tensor.backend().releaseTemporaryResources();
        if (!Float.isFinite(sum)) throw new IllegalStateException("Non-finite result");
        return (System.nanoTime() - started) / 1_000_000_000.0;
    }

    static Tensor product(Tensor a, Tensor b, boolean weightGradient, boolean flagged) {
        TensorBackend backend = Tensor.backend();
        return flagged
                ? backend.batchedMatmul(a, b, 1, weightGradient, !weightGradient)
                : backend.matmul(weightGradient ? backend.transpose(a) : a,
                        weightGradient ? b : backend.transpose(b));
    }

    private static double median(double[] samples) {
        Arrays.sort(samples);
        return (samples[9] + samples[10]) / 2;
    }
}
