package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.function.Function;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class UnaryGradientTest {
    @BeforeEach
    void useCpu() { Tensor.setBackend(new CpuBackend()); }

    public static Stream<Function<Tensor, Tensor>> operations() {
        return Stream.of(x -> x.addScalar(2), x -> x.multiplyScalar(1.3f), x -> x.divideScalar(2),
                Tensor::transpose, Tensor::sqrt, Tensor::neg, Tensor::exp, Tensor::log,
                x -> x.pow(2.3f), x -> x.pow(0), Tensor::tanhActivation, Tensor::sigmoidActivation,
                Tensor::reluActivation, Tensor::geluActivation, Tensor::sumRows, Tensor::sumAlongRows,
                Tensor::sumAlongCols, Tensor::meanAlongRows, Tensor::varianceAlongRows, Tensor::sumTensor,
                Tensor::maxAlongRows, x -> x.clamp(0.4f, 2.5f),
                x -> x.softmaxRows().multiply(x),
                x -> ActivationOps.causalSoftmax(x, 3, 0.7f).multiply(x),
                x -> Tensor.sliceRows(x, new int[]{1, 0, 1}, x.cols), x -> x.getRow(1), Tensor::copy);
    }

    @ParameterizedTest
    @MethodSource("operations")
    void operatorDerivativesMatchIndependentFiniteDifferences(Function<Tensor, Tensor> operation) {
        OperationGradientAssertions.check(operation, point());
    }

    @Test
    void maximumSplitsTiesAndClippingDefinesBoundaryGradients() {
        Tensor maximum = Tensor.from2D(new float[][]{{2, 2, 1}}).requiresGrad(true);
        maximum.maxAlongRows().backward();
        assertArrayEquals(new float[]{0.5f, 0.5f, 0}, maximum.grad().data);
        Tensor clipping = Tensor.from2D(new float[][]{{-2, -1, 0, 1, 2}}).requiresGrad(true);
        clipping.clamp(-1, 1).sumTensor().backward();
        assertArrayEquals(new float[]{0, 0, 1, 0, 0}, clipping.grad().data);
    }

    private static Tensor point() {
        return Tensor.from2D(new float[][]{{0.2f, 0.7f, 1.3f}, {1.9f, 2.3f, 3.1f}, {0.5f, 1.1f, 2.7f}});
    }
}
