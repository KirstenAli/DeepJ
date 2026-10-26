package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertTensorGradients;
import static org.junit.jupiter.api.Assertions.*;

class MetalNdIndexGradientTest extends MetalDifferentialTestSupport {
    @Test
    void stridedRepeatedSelectionMatchesActualGpuFiniteDifferences() {
        Tensor input = Tensor.from(new float[]{0.1f, -0.2f, 0.3f, 0.4f, 0.5f, -0.6f}, 1, 3, 2);
        Tensor upstream = Tensor.from(new float[]{0.2f, -0.1f, 0.5f, 0.3f, -0.2f, 0.4f}, 2, 3, 1);
        assertTensorGradients(input, value -> value.multiplyScalar(2).permute(2, 1, 0).indexSelect(1, 2, 0, 2),
                upstream, 1e-3f, 2e-4f, 1e-3f);
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void leafAccumulationReusesTheGradientAndSnapshotsCapturedIndices(DType dtype) {
        Tensor input = Tensor.from(new float[]{1, 2, 3}, dtype, 1, 3).requiresGrad(true);
        int[] indices = {2, 0, 2};
        Tensor output = input.indexSelect(1, indices).sumTensor();
        indices[0] = 99;
        output.backward();
        Tensor first = input.grad();
        input.indexSelect(1, 1).sumTensor().backward();
        assertSame(first, input.grad());
        assertEquals(DType.FLOAT32, input.grad().dtype());
        assertArrayEquals(new float[]{1, 1, 2}, input.grad().toFloatArray());
    }

    @Test
    void aliasedUpstreamAndSavedGradientBuffersKeepIndependentValues() {
        Tensor input = Tensor.from(new float[]{3, 4}).requiresGrad(true);
        Tensor upstream = Tensor.from(new float[]{1, 2});
        input.gradientState().bindGradient(upstream);
        input.indexSelect(0, 1, 0).backward(upstream);
        assertArrayEquals(new float[]{3, 3}, input.grad().toFloatArray());
        Tensor savedGradient = input.grad().requiresGrad(true);
        Tensor saved = savedGradient.pow(2).sumTensor();
        input.indexSelect(0, 0).sumTensor().backward();
        assertNotSame(savedGradient, input.grad());
        assertArrayEquals(new float[]{3, 3}, savedGradient.toFloatArray());
        saved.backward();
        assertArrayEquals(new float[]{6, 6}, savedGradient.grad().toFloatArray());
    }

    @Test
    void legacyCompactRowGatherAndScatterUseTheTypedGpuPath() {
        Tensor input = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, DType.FLOAT16, 3, 2).requiresGrad(true);
        Tensor.sliceRows(input, new int[]{2, 0, 2}, 2).sumTensor().backward();
        assertArrayEquals(new float[]{1, 1, 0, 0, 2, 2}, input.grad().toFloatArray());
        Tensor empty = new Tensor(DType.BFLOAT16, 0, 2).requiresGrad(true);
        empty.indexSelect(0).sumTensor().backward();
        assertArrayEquals(new int[]{0, 2}, empty.grad().shape());
    }
}
