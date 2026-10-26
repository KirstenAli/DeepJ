package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.CompactFloatReference;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

class MetalNdReductionReferenceTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(DType.class)
    void randomizedAxisPartitionsMatchIndependentCoordinateAndFp64Statistics(DType dtype) {
        Random random = new Random(20261026);
        for (int rank = 1; rank <= 6; rank++) {
            int[] dimensions = new int[rank];
            Arrays.fill(dimensions, 2);
            float[] values = new float[1 << rank];
            for (int index = 0; index < values.length; index++) { values[index] = (random.nextInt(64) - 32) / 16.0f; }
            Tensor input = Tensor.from(values, dtype, dimensions);
            int[] reverse = IntStream.range(0, rank).map(axis -> input.rank() - axis - 1).toArray();
            Tensor permuted = input.permute(reverse);
            int[] axes = IntStream.range(0, rank).filter(axis -> axis % 2 == 0).toArray();
            compare(permuted, axes);
        }
    }

    private void compare(Tensor input, int[] axes) {
        boolean[] selected = new boolean[input.rank()];
        for (int axis : axes) { selected[axis] = true; }
        double[][] statistics = statistics(input, selected);
        Tensor[] results = {input.sumAxes(axes), input.meanAxes(axes),
                input.varianceAxes(axes, false, 0), input.varianceAxes(axes, false, 1), input.maxAxes(axes)};
        int count = 1 << axes.length;
        for (int operation = 0; operation < results.length; operation++) {
            float[] expected = expected(statistics, count, operation, input.dtype());
            assertArrayEquals(expected, results[operation].toFloatArray(), 3e-6f);
        }
    }

    private static double[][] statistics(Tensor input, boolean[] selected) {
        int retained = 0;
        for (boolean axis : selected) { if (!axis) retained++; }
        double[][] statistics = new double[3][1 << retained];
        Arrays.fill(statistics[2], Double.NEGATIVE_INFINITY);
        for (int index = 0; index < input.numel(); index++) {
            int[] coordinates = coordinates(input.rank(), index);
            int group = group(coordinates, selected);
            double value = input.get(coordinates);
            statistics[0][group] += value;
            statistics[1][group] += value * value;
            statistics[2][group] = Math.max(statistics[2][group], value);
        }
        return statistics;
    }

    private static int[] coordinates(int rank, int index) {
        int[] coordinates = new int[rank];
        for (int axis = rank - 1; axis >= 0; axis--) {
            coordinates[axis] = index % 2;
            index /= 2;
        }
        return coordinates;
    }

    private static int group(int[] coordinates, boolean[] selected) {
        int group = 0;
        for (int axis = 0; axis < coordinates.length; axis++) {
            if (!selected[axis]) group = group * 2 + coordinates[axis];
        }
        return group;
    }

    private static float[] expected(double[][] statistics, int count, int operation, DType dtype) {
        float[] expected = new float[statistics[0].length];
        for (int group = 0; group < expected.length; group++) {
            double sum = statistics[0][group], squares = statistics[1][group];
            double value = switch (operation) {
                case 0 -> sum;
                case 1 -> sum / count;
                case 2 -> (squares - sum * sum / count) / count;
                case 3 -> (squares - sum * sum / count) / (count - 1);
                default -> statistics[2][group];
            };
            expected[group] = quantize((float) value, dtype);
        }
        return expected;
    }

    private static float quantize(float value, DType dtype) {
        return switch (dtype) {
            case FLOAT32 -> value;
            case FLOAT16 -> CompactFloatReference.half(Float.floatToFloat16(value) & 0xffff);
            case BFLOAT16 -> Float.intBitsToFloat((CompactFloatReference.bfloat(value) & 0xffff) << 16);
        };
    }
}
