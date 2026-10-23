package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;

class SavedGradientBufferTest {
    @BeforeEach
    void useCpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void accumulationPreservesGradientStorageNeededByAnotherRule(boolean gather) {
        Parameter parameter = new Parameter(Tensor.ones(2, 1));
        Tensor saved = Tensor.from2D(new float[][]{{2}, {4}});
        parameter.setGradient(saved);
        Tensor input = Tensor.ones(2, 1).requiresGrad(true);
        Tensor leaf = gather ? Tensor.sliceRows(parameter.value, new int[]{0, 1}, 1)
                : parameter.value.multiplyScalar(3);
        leaf.sumTensor().add(input.multiply(saved).sumTensor()).backward();
        float addition = gather ? 1 : 3;
        assertArrayEquals(new float[]{2, 4}, input.grad().data);
        assertArrayEquals(new float[]{2, 4}, saved.data);
        assertArrayEquals(new float[]{2 + addition, 4 + addition}, parameter.grad().data);
        assertNotSame(saved, parameter.grad());
    }
}
