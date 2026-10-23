package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MetalTransferTest extends MetalDifferentialTestSupport {
    private static final int[] BITS = {0, 0x80000000, 1, 0x7f7fffff,
            0x7f800000, 0xff800000, 0x7fc01234, 0xffc05678};

    @ParameterizedTest
    @ValueSource(ints = {1, 17, 65_537, 1_048_576})
    void repeatedTransfersPreserveEveryFloatBitAndSnapshot(int count) {
        Tensor tensor = new Tensor(1, count);
        var buffer = metal.graph.ensureGpuBuffer(tensor);
        metal.graph.flush();
        float[] actual = new float[count];
        for (int round = 0; round < 4; round++) {
            float[] input = values(count, round);
            MetalNative.nativeUploadBuffer(buffer.id, input);
            Arrays.fill(input, 17f);
            Arrays.fill(actual, 42f);
            MetalNative.nativeDownloadBuffer(buffer.id, actual);
            assertBits(actual, round);
        }
    }

    private static float[] values(int count, int round) {
        float[] values = new float[count];
        for (int index = 0; index < count; index++) {
            values[index] = Float.intBitsToFloat(bits(index, round));
        }
        return values;
    }

    private static void assertBits(float[] actual, int round) {
        for (int index = 0; index < actual.length; index++) {
            assertEquals(bits(index, round), Float.floatToRawIntBits(actual[index]));
        }
    }

    private static int bits(int index, int round) {
        return BITS[(index + round) % BITS.length];
    }
}
