package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorTestStorage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class MetalNdReductionTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(DType.class)
    void rankZeroThroughSixAxisStatisticsMatchStoredValueCpuReferences(DType dtype) {
        for (int rank = 0; rank <= 6; rank++) {
            int[] shape = new int[rank];
            Arrays.fill(shape, 2);
            float[] values = new float[1 << rank];
            for (int index = 0; index < values.length; index++) { values[index] = (index - 7) * 0.125f; }
            Tensor input = Tensor.from(values, dtype, shape);
            compare(input, new int[0], false);
            if (rank != 0) compare(input, new int[]{rank - 1}, true);
        }
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void groupedReductionsSupportUnsortedAxesOffsetsAndBroadcastStrides(DType dtype) {
        Tensor input = Tensor.from(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, dtype, 2, 2, 3);
        Tensor view = input.permute(2, 0, 1).slice(0, 0, 3, 2);
        compare(view, new int[]{-1, 0}, false);
        compare(view, new int[]{0, -1}, true);
        compare(input.select(0, 0).unsqueeze(0).expand(4, 2, 3), new int[]{0, 2}, false);
    }

    private void compare(Tensor input, int[] axes, boolean keep) {
        Tensor sum = metal.sumAxes(input, axes, keep), mean = metal.meanAxes(input, axes, keep);
        assertResult(cpu.sumAxes(input, axes, keep), sum);
        assertResult(cpu.meanAxes(input, axes, keep), mean);
        assertResult(cpu.varianceAxes(input, axes, keep, 0), metal.varianceAxes(input, axes, keep, 0));
        assertResult(cpu.maximumAxes(input, axes, keep), metal.maximumAxes(input, axes, keep));
    }

    private static void assertResult(Tensor expected, Tensor actual) {
        assertFalse(TensorTestStorage.hasHostStorage(actual));
        assertEquals(expected.dtype(), actual.dtype());
        assertArrayEquals(expected.shape(), actual.shape());
        assertArrayEquals(expected.toFloatArray(), actual.toFloatArray(), 3e-6f);
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void emptyAndUndefinedStatisticsRetainShapeAndDoNotUploadEmptyInputs(DType dtype) {
        Tensor input = new Tensor(dtype, 2, 0, 3);
        assertArrayEquals(new float[6], input.sum(1, false).toFloatArray());
        assertAllNan(input.mean(1, false));
        assertAllNan(input.variance(1, false));
        assertThrows(IllegalArgumentException.class, () -> input.maximum(1, false));
        assertEquals(0, input.maximum(0, false).numel());
        assertNull(input.getGpuTag());
        Tensor singleton = Tensor.from(new float[]{1, 2}, dtype, 2, 1);
        assertAllNan(singleton.variance(1, true, 1));
        assertEquals(0, metal.zeros(0, 3).numel());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void maximumPreservesSignedZerosAndPropagatesNanValues(DType dtype) {
        Tensor input = Tensor.from(new float[]{-0.0f, 0.0f, -0.0f, -0.0f}, dtype, 2, 2);
        float[] maximum = input.maximum(1, false).toFloatArray();
        assertEquals(0, Float.floatToRawIntBits(maximum[0]));
        assertEquals(0x80000000, Float.floatToRawIntBits(maximum[1]));
        assertAllNan(Tensor.from(new float[]{1, Float.NaN, 2}, dtype).maxAxes());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void totalsAndNormsAccumulateFloat32InsteadOfRoundingCompactIntermediates(DType dtype) {
        Tensor input = Tensor.from(new float[]{2048, 1, 1, -2}, dtype, 1, 2, 2);
        assertEquals(cpu.sum(input), metal.sum(input));
        assertEquals(cpu.sumAbs(input), metal.sumAbs(input));
        assertEquals(cpu.l2Norm(List.of(input)), metal.l2Norm(List.of(input)), 3e-6f);
        assertResult(cpu.meanAxes(input, new int[]{2}, true), input.mean(2, true));
    }

    @Test
    void reductionsLeaveLazyGpuSourcesUnmaterializedAndKeepOnlyRealTypedOutputStorage() {
        Tensor source = metal.addScalar(metal.zeros(3, 4), 2).to(DType.BFLOAT16).unsqueeze(0);
        Tensor variance = source.variance(-1, false), mean = source.mean(1, false);
        assertArrayEquals(new float[]{0, 0, 0}, variance.toFloatArray());
        assertArrayEquals(new float[]{2, 2, 2, 2}, mean.toFloatArray());
        assertFalse(TensorTestStorage.hasHostStorage(source));
        assertEquals(6, variance.storageBytes());
        assertEquals(8, mean.storageBytes());
    }

    private static void assertAllNan(Tensor tensor) {
        for (float value : tensor.toFloatArray()) { assertTrue(Float.isNaN(value)); }
    }
}
