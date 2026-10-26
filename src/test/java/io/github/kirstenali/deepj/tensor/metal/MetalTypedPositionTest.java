package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.ActivationOps;
import io.github.kirstenali.deepj.autograd.LayoutOps;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.testing.PositionReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertTensorGradients;
import static org.junit.jupiter.api.Assertions.*;

class MetalTypedPositionTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(DType.class)
    void stridedTypedPositionsMatchIndependentQuantizedEquationsWithoutHostShadows(DType dtype) {
        Tensor input = resident(PositionReference.input(dtype));
        Tensor cosine = resident(PositionReference.table(DType.FLOAT32, false));
        Tensor sine = resident(PositionReference.table(dtype, true));
        Tensor mask = metal.causalMask(input, 4), probabilities = metal.causalSoftmax(input, 4, 0.625f);
        Tensor rotated = metal.rotary(input, cosine, sine, 4, false);
        assertFalse(TensorTestStorage.hasHostStorage(mask));
        assertFalse(TensorTestStorage.hasHostStorage(probabilities));
        assertFalse(TensorTestStorage.hasHostStorage(rotated));
        assertArrayEquals(PositionReference.mask(input, 4), mask.toFloatArray());
        assertArrayEquals(PositionReference.softmax(input, 4, 0.625f), probabilities.toFloatArray(), 3e-6f);
        assertArrayEquals(PositionReference.rotary(input, cosine, sine, 4, false), rotated.toFloatArray(), 3e-6f);
        assertEquals(dtype, rotated.dtype());
        assertEquals(dtype.storageBytes(input.numel()), rotated.storageBytes());
    }

    @Test
    void actualGpuFiniteDifferencesCoverGenericCausalAndRotaryAdjoints() {
        Tensor input = PositionReference.input(DType.FLOAT32);
        Tensor cosine = PositionReference.table(DType.FLOAT32, false);
        Tensor sine = PositionReference.table(DType.FLOAT32, true);
        assertTensorGradients(input.permute(1, 0), x -> ActivationOps.causalSoftmax(x.permute(1, 0), 4, 0.625f),
                input.detach(), 1e-3f, 3e-4f, 2e-3f);
        assertTensorGradients(input, x -> LayoutOps.rotary(x, cosine, sine, 4, false),
                input.detach(), 1e-3f, 3e-4f, 2e-3f);
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void futureNonfiniteValuesAreSkippedAndEmptyInputsNeedNoDeviceAllocation(DType dtype) {
        Tensor source = Tensor.from(new float[]{0, 0, Float.NaN, 1}, dtype, 2, 2).permute(1, 0);
        Tensor output = metal.graph.layoutCausal(source, 2, 1, 2);
        assertArrayEquals(new float[]{1, 0}, new float[]{output.get(0, 0), output.get(0, 1)});
        Tensor empty = metal.causalSoftmax(new Tensor(dtype, 0, 4), 4, 1);
        assertEquals(0, empty.numel());
        assertNull(empty.getGpuTag());
        assertEquals(0, metal.rotary(new Tensor(dtype, 4, 0), new Tensor(4, 0), new Tensor(4, 0), 4, false).numel());
    }

    private Tensor resident(Tensor strided) {
        return metal.copy(strided.permute(1, 0)).permute(1, 0);
    }
}
