package io.github.kirstenali.deepj.onnx;

import static org.junit.jupiter.api.Assertions.*;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

class ValueContractsTest {
    @BeforeEach
    void cpu() {
        Tensor.setBackend(new CpuBackend());
    }

    @ParameterizedTest
    @CsvSource({
        "2,0,255",
        "3,-128,127",
        "4,0,65535",
        "5,-32768,32767",
        "6,-2147483648,2147483647",
        "9,0,1",
        "12,0,4294967295"
    })
    void exactIntegerInputsAcceptBoundariesAndRejectOverflow(int type, long minimum, long maximum) {
        var value = OnnxValue.integerTensor(type, new long[] {minimum, maximum}, 2);
        assertArrayEquals(new long[] {minimum, maximum}, value.longs());
        assertThrows(
                IllegalArgumentException.class,
                () -> OnnxValue.integerTensor(type, new long[] {minimum - 1}, 1));
        assertThrows(
                IllegalArgumentException.class,
                () -> OnnxValue.integerTensor(type, new long[] {maximum + 1}, 1));
    }

    @Test
    void inputArraysAndShapesAreDefensivelyCopied() {
        long[] values = {7, 8};
        int[] shape = {2};
        var value = OnnxValue.integers(values, shape);
        values[0] = -1;
        shape[0] = 1;
        value.longs()[0] = -2;
        value.shape()[0] = 3;
        assertArrayEquals(new long[] {7, 8}, value.longs());
        assertArrayEquals(new int[] {2}, value.shape());
    }

    @Test
    void rejectsInvalidShapesAndTypedAccess() {
        assertThrows(IllegalArgumentException.class, () -> OnnxValue.integers(new long[] {1}, 2));
        assertThrows(IllegalArgumentException.class, () -> OnnxValue.integers(new long[0], 0, -1));
        assertThrows(
                ArithmeticException.class,
                () -> OnnxValue.integers(new long[0], Integer.MAX_VALUE, 2));
        assertThrows(
                IllegalArgumentException.class,
                () -> OnnxValue.integers(new long[] {1}, 1).tensor());
        assertThrows(IllegalArgumentException.class, () -> OnnxValue.of(Tensor.ones(1, 1)).longs());
        assertThrows(
                IllegalArgumentException.class,
                () -> OnnxValue.integers(new long[] {1, 2}, 2).scalar());
        assertThrows(
                IllegalArgumentException.class,
                () -> OnnxValue.integerTensor(1, new long[] {1}, 1));
        assertThrows(
                ArithmeticException.class,
                () -> OnnxValue.integers(new long[] {Long.MAX_VALUE}, 1).ints());
    }
}
