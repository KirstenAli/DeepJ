package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Random;
import java.util.function.BiFunction;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class BinaryGradientTest {
    @BeforeEach
    void useCpu() { Tensor.setBackend(new CpuBackend()); }

    static Stream<BiFunction<Tensor, Tensor, Tensor>> elementwise() {
        return Stream.of(Tensor::add, Tensor::subtract, Tensor::multiply, Tensor::divide, ActivationOps::swiGlu);
    }

    @ParameterizedTest
    @MethodSource("elementwise")
    void bothElementwiseOperandsMatchNumericalGradients(BiFunction<Tensor, Tensor, Tensor> operation) {
        checkBoth(operation, point(2, 3), point(2, 3));
    }

    static Stream<BiFunction<Tensor, Tensor, Tensor>> rowBroadcasts() {
        return Stream.of(Tensor::addRowVector, Tensor::addBroadcastRows, Tensor::multiplyBroadcastRows);
    }

    @ParameterizedTest
    @MethodSource("rowBroadcasts")
    void rowBroadcastsReduceTheBroadcastOperandGradient(BiFunction<Tensor, Tensor, Tensor> operation) {
        checkBoth(operation, point(2, 3), point(1, 3));
    }

    static Stream<BiFunction<Tensor, Tensor, Tensor>> columnBroadcasts() {
        return Stream.of(Tensor::addBroadcastCols, Tensor::subtractBroadcastCols,
                Tensor::multiplyBroadcastCols, Tensor::divideBroadcastCols);
    }

    @ParameterizedTest
    @MethodSource("columnBroadcasts")
    void columnBroadcastsReduceTheBroadcastOperandGradient(BiFunction<Tensor, Tensor, Tensor> operation) {
        checkBoth(operation, point(2, 3), point(2, 1));
    }

    @Test
    void rectangularMatrixMultiplicationMatchesFiniteDifferences() {
        checkBoth(Tensor::matmul, point(2, 3), point(3, 4));
    }

    @Test
    void oneTrackedOperandAndConstantOperandsUseTheSameDerivativeRules() {
        Tensor left = point(2, 3), right = point(3, 2);
        assertEquals(false, left.matmul(right).requiresGrad());
        OperationGradientAssertions.check(x -> x.matmul(right), left);
        OperationGradientAssertions.check(x -> left.matmul(x), right);
    }

    private static void checkBoth(BiFunction<Tensor, Tensor, Tensor> operation, Tensor left, Tensor right) {
        OperationGradientAssertions.check(x -> operation.apply(x, right), left);
        OperationGradientAssertions.check(x -> operation.apply(left, x), right);
    }

    private static Tensor point(int rows, int columns) {
        return Tensor.random(rows, columns, new Random(123)).addScalar(1);
    }
}
