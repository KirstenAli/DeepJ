package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/** Independent logical-coordinate selection and chronological FLOAT32 update references. */
class MetalNdIndexReferenceTest extends MetalDifferentialTestSupport {
    @Test
    void everyAxisOfRanksOneThroughSixMatchesCoordinateReferencesForAllDtypes() {
        for (DType dtype : DType.values()) {
            for (int rank = 1; rank <= 6; rank++) {
                for (int axis = 0; axis < rank; axis++) { checkAxis(dtype, rank, axis); }
            }
        }
    }

    private static void checkAxis(DType dtype, int rank, int axis) {
        Tensor source = source(dtype, rank);
        int[] shape = source.shape();
        float[] original = source.toFloatArray();
        int[] indices = {shape[axis] - 1, 0, shape[axis] - 1, 0};
        int[] selectedShape = shape.clone();
        selectedShape[axis] = indices.length;
        float[] selected = selected(original, shape, selectedShape, axis, indices);
        assertArrayEquals(selected, source.indexSelect(axis, indices).toFloatArray());
        Tensor updates = Tensor.from(selected, dtype, selectedShape), target = new Tensor(source);
        target.indexAddInPlace(axis, indices, updates);
        assertArrayEquals(accumulated(original, shape, selected, selectedShape, axis, indices, dtype),
                target.toFloatArray());
    }

    private static Tensor source(DType dtype, int rank) {
        int[] shape = new int[rank];
        Arrays.fill(shape, 2);
        shape[rank - 1] = 3;
        float[] values = new float[IntStream.of(shape).reduce(1, Math::multiplyExact)];
        for (int index = 0; index < values.length; index++) { values[index] = (index % 13 - 6) / 8.0f; }
        return Tensor.from(values, dtype, shape).permute(IntStream.range(0, rank).map(i -> rank - i - 1).toArray());
    }

    private static float[] selected(float[] original, int[] shape, int[] selectedShape, int axis, int[] indices) {
        float[] result = new float[IntStream.of(selectedShape).reduce(1, Math::multiplyExact)];
        for (int index = 0; index < result.length; index++) {
            result[index] = original[destination(index, shape, selectedShape, axis, indices)];
        }
        return result;
    }

    private static float[] accumulated(float[] original, int[] shape, float[] updates, int[] selectedShape,
                                       int axis, int[] indices, DType dtype) {
        float[] result = original.clone();
        for (int index = 0; index < updates.length; index++) {
            result[destination(index, shape, selectedShape, axis, indices)] += updates[index];
        }
        return Tensor.from(result, dtype, shape).toFloatArray();
    }

    private static int destination(int index, int[] shape, int[] selectedShape, int selected, int[] indices) {
        int result = 0, stride = 1;
        for (int axis = shape.length - 1; axis >= 0; axis--) {
            int coordinate = index % selectedShape[axis];
            index /= selectedShape[axis];
            result += (axis == selected ? indices[coordinate] : coordinate) * stride;
            stride *= shape[axis];
        }
        return result;
    }
}
