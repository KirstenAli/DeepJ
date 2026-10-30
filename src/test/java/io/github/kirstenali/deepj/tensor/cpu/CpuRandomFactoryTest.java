package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class CpuRandomFactoryTest {
    private final CpuBackend backend = new CpuBackend();

    @Test
    void everySeededValueAndFollowingRandomStateMatchIndependentReference() {
        for (long seed : new long[]{0, 1, 2026, Long.MIN_VALUE}) {
            for (int[] shape : new int[][]{{1, 1}, {2, 3}, {11, 17}}) {
                checkSeed(seed, shape[0], shape[1]);
            }
        }
    }

    private void checkSeed(long seed, int rows, int columns) {
        Random actual = new Random(seed), reference = new Random(seed);
        Tensor result = backend.random(rows, columns, actual);
        assertArrayEquals(new int[]{rows, columns}, result.shape());
        assertEquals(DType.FLOAT32, result.dtype());
        for (float value : result.rawData()) {
            assertBits((float) (reference.nextGaussian() * .1), value);
        }
        assertEquals(reference.nextLong(), actual.nextLong());
    }

    @ParameterizedTest
    @CsvSource({"0,0", "0,7", "7,0"})
    void emptyAxesDoNotConsumeTheRandomSource(int rows, int columns) {
        Sequence random = new Sequence(0, -0.0, 1);
        Tensor result = backend.random(rows, columns, random);
        assertArrayEquals(new int[]{rows, columns}, result.shape());
        assertEquals(0, result.numel());
        assertEquals(0, random.calls);
    }

    @ParameterizedTest
    @CsvSource({"-1,2", "2,-1", "2147483647,2"})
    void invalidShapesFailBeforeConsumingTheRandomSource(int rows, int columns) {
        Sequence random = new Sequence(1);
        assertThrows(IllegalArgumentException.class, () -> backend.random(rows, columns, random));
        assertEquals(0, random.calls);
    }

    @Test
    void customGaussianValuesPreserveSignedZerosNonfiniteValuesAndCallOrder() {
        double[] values = {0.0, -0.0, 1.0, -1.0, Double.MIN_VALUE, Double.MAX_VALUE,
                Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, .25};
        Sequence random = new Sequence(values);
        Tensor result = backend.random(2, 5, random);
        for (int index = 0; index < values.length; index++) {
            assertBits((float) (values[index] * .1), result.rawData()[index]);
        }
        assertEquals(values.length, random.calls);
    }

    @Test
    void callbackFailurePropagatesTheExactExceptionWithoutExtraCalls() {
        var random = new FailingRandom();
        assertSame(random.failure, assertThrows(IllegalStateException.class, () -> backend.random(2, 3, random)));
        assertEquals(3, random.calls);
    }

    @Test
    void emptyFactoriesDoNotDereferenceANullRandomSource() {
        assertEquals(0, backend.random(0, 7, null).numel());
        assertEquals(0, backend.random(7, 0, null).numel());
    }

    @Test
    void nonemptyFactoriesRejectANullRandomSource() {
        assertThrows(NullPointerException.class, () -> backend.random(1, 1, null));
    }

    private static void assertBits(float expected, float actual) {
        assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(actual));
    }

    private static final class Sequence extends Random {
        private final double[] values;
        private int calls;

        private Sequence(double... values) { this.values = values; }

        @Override public double nextGaussian() { return values[calls++ % values.length]; }
    }

    private static final class FailingRandom extends Random {
        private final IllegalStateException failure = new IllegalStateException("Gaussian callback");
        private int calls;

        @Override public double nextGaussian() {
            if (++calls == 3) throw failure;
            return .25;
        }
    }
}
