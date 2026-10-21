package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.function.UnaryOperator;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertSame;

class MetalInPlaceTest extends MetalDifferentialTestSupport {

    @ParameterizedTest
    @MethodSource({"arithmetic", "activations"})
    void tensorInPlaceOperationsPreserveIdentityAndMatchCpu(UnaryOperator<Tensor> operation) {
        Tensor expected = positive(3, 7, 91L);
        Tensor actual = new Tensor(expected);
        Tensor.setBackend(cpu);
        assertSame(expected, operation.apply(expected));
        Tensor.setBackend(metal);
        assertSame(actual, operation.apply(actual));
        assertTensorClose(expected, actual, 1e-4f, 1e-4f);
    }

    private static Stream<UnaryOperator<Tensor>> arithmetic() {
        Tensor other = Tensor.from2D(new float[][]{{2, 3, 4, 5, 6, 7, 8},
                {3, 4, 5, 6, 7, 8, 9}, {4, 5, 6, 7, 8, 9, 10}});
        return Stream.of(value -> value.addInPlace(new Tensor(other)),
                value -> value.subtractInPlace(new Tensor(other)),
                value -> value.multiplyInPlace(new Tensor(other)),
                value -> value.divideInPlace(new Tensor(other)),
                value -> value.addScalarInPlace(2), value -> value.multiplyScalarInPlace(2),
                value -> value.divideScalarInPlace(2));
    }

    private static Stream<UnaryOperator<Tensor>> activations() {
        return Stream.of(Tensor::sqrtInPlace, Tensor::negInPlace, Tensor::expInPlace,
                Tensor::logInPlace, Tensor::reluInPlace, Tensor::geluInPlace,
                Tensor::tanhInPlace, Tensor::sigmoidInPlace);
    }
}
