package io.github.kirstenali.deepj.onnx;

import static org.junit.jupiter.api.Assertions.*;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

class ShapeContractsTest {
    @BeforeEach
    void cpu() {
        Tensor.setBackend(new CpuBackend());
    }

    @ParameterizedTest
    @MethodSource("invalidReshapes")
    void rejectsAmbiguousOrImpossibleReshapes(long[] shape) {
        OnnxValue[] inputs = {source(), OnnxValue.integers(shape, shape.length)};
        assertThrows(IllegalArgumentException.class, () -> OperatorCalls.run("Reshape", inputs));
    }

    static Stream<long[]> invalidReshapes() {
        return Stream.of(
                new long[] {-1, -1}, new long[] {-2, 3}, new long[] {4, -1}, new long[] {0, 0, 0});
    }

    @Test
    void reshapePreservesValuesWithCopiedAndInferredDimensions() {
        var result =
                OperatorCalls.run(
                                "Reshape",
                                new OnnxValue[] {
                                    source(), OnnxValue.integers(new long[] {0, -1}, 2)
                                })
                        .get(0);
        assertArrayEquals(new int[] {2, 3}, result.shape());
        assertArrayEquals(new float[] {1, 2, 3, 4, 5, 6}, result.tensor().toFloatArray());
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        OperatorCalls.run(
                                "Reshape",
                                new OnnxValue[] {
                                    source(), OnnxValue.integers(new long[] {0, -1}, 2)
                                },
                                TestModels.integer("allowzero", 1)));
    }

    @ParameterizedTest
    @ValueSource(ints = {-3, 3})
    void flattenRejectsOutOfRangeAxes(int axis) {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        OperatorCalls.run(
                                "Flatten",
                                new OnnxValue[] {source()},
                                TestModels.integer("axis", axis)));
    }

    @Test
    void flattenSupportsNegativeAndBoundaryAxes() {
        for (int axis : new int[] {-1, 0, 2}) {
            var value =
                    OperatorCalls.run(
                                    "Flatten",
                                    new OnnxValue[] {source()},
                                    TestModels.integer("axis", axis))
                            .get(0);
            assertEquals(6, value.size());
            assertEquals(2, value.shape().length);
        }
    }

    @Test
    void squeezeChecksSingletonAxesAndUnsqueezeRequiresUniqueAxes() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        OperatorCalls.run(
                                "Squeeze",
                                new OnnxValue[] {source(), OnnxValue.integers(new long[] {1}, 1)}));
        assertThrows(
                IllegalArgumentException.class,
                () -> OperatorCalls.run("Unsqueeze", new OnnxValue[] {source()}));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        OperatorCalls.run(
                                "Unsqueeze",
                                new OnnxValue[] {
                                    source(), OnnxValue.integers(new long[] {0, 0}, 2)
                                }));
    }

    @Test
    void squeezeAndUnsqueezePreserveFloatingValues() {
        var value = OnnxValue.of(Tensor.from(new float[] {1, 2}, 1, 2, 1));
        var squeezed = OperatorCalls.run("Squeeze", new OnnxValue[] {value}).get(0);
        assertArrayEquals(new int[] {2}, squeezed.shape());
        var expanded =
                OperatorCalls.run(
                                "Unsqueeze",
                                new OnnxValue[] {
                                    squeezed, OnnxValue.integers(new long[] {0, -1}, 2)
                                })
                        .get(0);
        assertArrayEquals(value.shape(), expanded.shape());
        assertArrayEquals(value.tensor().toFloatArray(), expanded.tensor().toFloatArray());
    }

    private static OnnxValue source() {
        return OnnxValue.of(Tensor.from(new float[] {1, 2, 3, 4, 5, 6}, 2, 3));
    }
}
