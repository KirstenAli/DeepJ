package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

/** Independent coordinate enumeration, not production layout/broadcast helpers. */
class NdOperatorReferenceTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void randomRanksZeroThroughSixMatchIndependentBroadcastAndPermutationReferences() {
        Random random = new Random(20261026);
        for (int rank = 0; rank <= 6; rank++) {
            for (DType dtype : DType.values()) {
                for (int sample = 0; sample < 3; sample++) {
                    checkCase(rank, dtype, random, sample != 0);
                }
            }
        }
    }

    private static void checkCase(int rank, DType dtype, Random random, boolean permuted) {
        int[] original = IntStream.range(0, rank).map(axis -> random.nextInt(1, 4)).toArray();
        int[] axes = IntStream.range(0, rank).map(axis -> rank - axis - 1).toArray();
        float[] source = values(original, random);
        int[] shape = permuted ? IntStream.of(axes).map(axis -> original[axis]).toArray() : original;
        float[] logical = permuted ? permutation(source, original, shape, axes) : source;
        int[] rightShape = Arrays.copyOfRange(shape, random.nextInt(rank + 1), rank);
        for (int axis = 0; axis < rightShape.length; axis++) {
            if (random.nextBoolean()) rightShape[axis] = 1;
        }
        float[] rightValues = values(rightShape, random);
        Tensor left = Tensor.from(source, dtype, original);
        if (permuted) left = left.permute(axes);
        Tensor right = Tensor.from(rightValues, random.nextBoolean() ? dtype : DType.FLOAT32, rightShape);
        compare(left, right, logical, shape, rightValues, rightShape);
    }

    private static void compare(Tensor left, Tensor right, float[] a, int[] shape, float[] b, int[] rightShape) {
        float[] sums = new float[a.length], products = new float[a.length], differences = new float[a.length];
        List<int[]> positions = coordinates(shape);
        for (int index = 0; index < a.length; index++) {
            float value = b[broadcastIndex(positions.get(index), rightShape)];
            sums[index] = a[index] + value;
            products[index] = a[index] * value;
            differences[index] = a[index] - value;
        }
        assertArrayEquals(sums, left.add(right).toFloatArray());
        assertArrayEquals(products, left.multiply(right).toFloatArray());
        assertArrayEquals(differences, left.subtract(right).toFloatArray());
        assertArrayEquals(shape, left.add(right).shape());
        assertEquals(DType.promote(left.dtype(), right.dtype()), left.multiply(right).dtype());
    }

    private static int broadcastIndex(int[] coordinate, int[] shape) {
        int index = 0, leading = coordinate.length - shape.length;
        for (int axis = 0; axis < shape.length; axis++) {
            index = index * shape[axis] + (shape[axis] == 1 ? 0 : coordinate[axis + leading]);
        }
        return index;
    }

    private static float[] permutation(float[] values, int[] original, int[] shape, int[] axes) {
        float[] output = new float[values.length];
        List<int[]> coordinates = coordinates(shape);
        for (int index = 0; index < output.length; index++) {
            int[] source = new int[axes.length];
            for (int axis = 0; axis < axes.length; axis++) {
                source[axes[axis]] = coordinates.get(index)[axis];
            }
            output[index] = values[broadcastIndex(source, original)];
        }
        return output;
    }

    private static List<int[]> coordinates(int[] shape) {
        List<int[]> result = new ArrayList<>();
        enumerate(shape, new int[shape.length], 0, result);
        return result;
    }

    private static void enumerate(int[] shape, int[] coordinate, int axis, List<int[]> result) {
        if (axis == shape.length) {
            result.add(coordinate.clone());
            return;
        }
        for (int value = 0; value < shape[axis]; value++) {
            coordinate[axis] = value;
            enumerate(shape, coordinate, axis + 1, result);
        }
    }

    private static float[] values(int[] shape, Random random) {
        int count = IntStream.of(shape).reduce(1, Math::multiplyExact);
        float[] result = new float[count];
        for (int index = 0; index < count; index++) {
            result[index] = random.nextInt(-7, 8) / 8.0f;
        }
        return result;
    }
}
