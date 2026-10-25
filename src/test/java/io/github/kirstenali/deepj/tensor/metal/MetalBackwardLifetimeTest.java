package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.Autograd;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

class MetalBackwardLifetimeTest extends MetalDifferentialTestSupport {
    @Test
    void customBackwardCanKeepAnUpstreamCreatedByABuiltinDerivative() {
        Tensor input = Tensor.from2D(new float[][]{{2, 3}}).requiresGrad(true);
        AtomicReference<Tensor> captured = new AtomicReference<>();
        Tensor custom = Autograd.record(metal.addScalar(input, 0), gradient -> {
            captured.set(gradient);
            return new Tensor[]{gradient};
        }, input);
        custom.multiplyScalar(4).sumTensor().backward();
        assertTensorClose(Tensor.from2D(new float[][]{{4, 4}}), input.grad(), 0, 0);
        assertTensorClose(Tensor.from2D(new float[][]{{4, 4}}), captured.get(), 0, 0);
        assertTensorClose(Tensor.from2D(new float[][]{{2, 3}}), custom, 0, 0);
    }

    @Test
    void externallySuppliedUpstreamAndIntermediateOutputsRemainReadable() {
        Tensor input = Tensor.from2D(new float[][]{{2, 3}}).requiresGrad(true);
        Tensor intermediate = input.multiply(input);
        Tensor output = intermediate.multiplyScalar(3);
        Tensor upstream = Tensor.from2D(new float[][]{{4, 5}});
        output.backward(upstream);
        assertTensorClose(Tensor.from2D(new float[][]{{48, 90}}), input.grad(), 0, 0);
        assertTensorClose(Tensor.from2D(new float[][]{{4, 9}}), intermediate, 0, 0);
        assertTensorClose(Tensor.from2D(new float[][]{{4, 5}}), upstream, 0, 0);
        assertTensorClose(Tensor.from2D(new float[][]{{12, 27}}), output, 0, 0);
    }
}
