package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;

class MetalAutogradTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @MethodSource("io.github.kirstenali.deepj.autograd.UnaryGradientTest#operations")
    void trackedOperatorsMatchCpuOutputsAndGradients(Function<Tensor, Tensor> operation) {
        Tensor point = Tensor.from2D(new float[][]{{0.2f, 0.7f, 1.3f},
                {1.9f, 2.3f, 3.1f}, {0.5f, 1.1f, 2.7f}});
        Result expected = differentiate(cpu, operation, point);
        Result actual = differentiate(metal, operation, point);
        assertTensorClose(expected.output(), actual.output(), 2e-4f, 2e-3f);
        assertTensorClose(expected.gradient(), actual.gradient(), 3e-4f, 3e-3f);
    }

    private static Result differentiate(TensorBackend backend, Function<Tensor, Tensor> operation, Tensor point) {
        Tensor.setBackend(backend);
        Tensor input = new Tensor(point).requiresGrad(true);
        Tensor output = operation.apply(input);
        output.backward(Tensor.ones(output.rows, output.cols));
        return new Result(new Tensor(output), new Tensor(input.grad()));
    }

    @Test
    void maskedAndScaledCrossEntropyMatchCpu() {
        Tensor point = random(3, 5, 71);
        int[] targets = {1, 3, 2};
        boolean[] mask = {false, true, true};
        Function<Tensor, Tensor> loss = x -> CrossEntropyLoss.forward(x, targets, mask).multiplyScalar(2);
        Result expected = differentiate(cpu, loss, point);
        Result actual = differentiate(metal, loss, point);
        assertTensorClose(expected.output(), actual.output(), 2e-4f, 2e-3f);
        assertTensorClose(expected.gradient(), actual.gradient(), 3e-4f, 3e-3f);
    }

    @Test
    void temporaryCleanupBetweenForwardAndBackwardPreservesSavedDeviceValues() {
        Tensor input = Tensor.from2D(new float[][]{{2, 3}}).requiresGrad(true);
        Tensor output = input.multiply(input).tanhActivation();
        metal.releaseTemporaryResources();
        assertNotNull(input.getGpuTag());
        output.sumTensor().backward();
        Tensor expected = Tensor.zeros(input.rows, input.cols);
        input.grad().materialize();
        for (int index = 0; index < input.data.length; index++) {
            float value = input.data[index];
            float tangent = (float) Math.tanh(value * value);
            expected.data[index] = 2 * value * (1 - tangent * tangent);
        }
        assertTensorClose(expected, input.grad(), 1e-5f, 1e-3f);
    }

    @Test
    void noGradForwardDoesNotPinTemporaryDeviceBuffers() {
        Tensor parameter = Tensor.ones(2, 2).requiresGrad(true);
        Tensor output;
        try (var ignored = GradMode.noGrad()) {
            output = parameter.matmul(parameter);
        }
        assertFalse(output.requiresGrad());
        metal.releaseTemporaryResources();
        assertNull(output.getGpuTag());
    }

    @Test
    void leafGatherSharesTheCanonicalBufferWithDenseGradientBranches() {
        Parameter table = new Parameter(Tensor.ones(3, 2));
        Tensor buffer = table.grad();
        Tensor gathered = Tensor.sliceRows(table.value, new int[]{1, 1, 2}, 2);
        table.value.sumTensor().add(gathered.sumTensor()).backward();
        assertSame(buffer, table.grad());
        Tensor expected = Tensor.from2D(new float[][]{{1, 1}, {3, 3}, {2, 2}});
        assertTensorClose(expected, table.grad(), 1e-6f, 1e-6f);
    }

    @Test
    void repeatedLeafGathersAccumulateOnDeviceAndRespectZeroGrad() {
        Parameter table = new Parameter(Tensor.ones(3, 2));
        Tensor.sliceRows(table.value, new int[]{1, 1}, 2).sumTensor().backward();
        Tensor buffer = table.grad();
        Tensor.sliceRows(table.value, new int[]{1, 2}, 2).sumTensor().backward();
        assertSame(buffer, table.grad());
        Tensor expected = Tensor.from2D(new float[][]{{0, 0}, {3, 3}, {1, 1}});
        assertTensorClose(expected, table.grad(), 1e-6f, 1e-6f);
        table.zeroGrad();
        Tensor.sliceRows(table.value, new int[]{0}, 2).sumTensor().backward();
        Tensor reset = Tensor.from2D(new float[][]{{1, 1}, {0, 0}, {0, 0}});
        assertTensorClose(reset, table.grad(), 1e-6f, 1e-6f);
    }

    @Test
    void gatherProtectsAnUpstreamThatIsAlsoTheAccumulationBuffer() {
        Tensor table = Tensor.ones(2, 1).requiresGrad(true);
        Tensor upstream = Tensor.from2D(new float[][]{{1}, {2}});
        table.gradientState().bindGradient(upstream);
        Tensor.sliceRows(table, new int[]{1, 0}, 1).backward(upstream);
        Tensor expected = Tensor.from2D(new float[][]{{3}, {3}});
        assertTensorClose(expected, table.grad(), 1e-6f, 1e-6f);
    }

    private record Result(Tensor output, Tensor gradient) {}
}
