package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class RandomLayoutTest {
    @Test
    void permutedRanksZeroThroughSixMatchIndependentRowMajorIndexing() {
        var random = new Random(2026);
        for (int rank = 0; rank <= 6; rank++) {
            for (int sample = 0; sample < 30; sample++) {
                int[] dimensions = random.ints(rank, 1, 4).toArray();
                checkPermutation(dimensions, permutation(rank, random));
            }
        }
    }

    private static void checkPermutation(int[] dimensions, int[] axes) {
        var layout = LayoutTransforms.permute(TensorLayout.dense(new TensorShape(dimensions)), axes);
        assertTrue(layout.nonOverlapping());
        int[] outputDimensions = layout.shape().dimensions();
        for (int index = 0; index < layout.shape().elements(); index++) {
            int[] coordinate = coordinate(index, outputDimensions), original = new int[axes.length];
            for (int axis = 0; axis < axes.length; axis++) {
                original[axes[axis]] = coordinate[axis];
            }
            assertEquals(rowMajor(original, dimensions), layout.addressAt(index));
            assertEquals(rowMajor(original, dimensions), layout.address(coordinate));
        }
    }

    private static int rowMajor(int[] coordinate, int[] dimensions) {
        int index = 0;
        for (int axis = 0; axis < dimensions.length; axis++) {
            index = index * dimensions[axis] + coordinate[axis];
        }
        return index;
    }

    private static int[] coordinate(int index, int[] dimensions) {
        int[] result = new int[dimensions.length];
        for (int axis = dimensions.length - 1; axis >= 0; axis--) {
            result[axis] = index % dimensions[axis];
            index /= dimensions[axis];
        }
        return result;
    }

    private static int[] permutation(int rank, Random random) {
        int[] result = java.util.stream.IntStream.range(0, rank).toArray();
        for (int end = rank - 1; end > 0; end--) {
            int index = random.nextInt(end + 1), value = result[end];
            result[end] = result[index];
            result[index] = value;
        }
        return result;
    }
}
