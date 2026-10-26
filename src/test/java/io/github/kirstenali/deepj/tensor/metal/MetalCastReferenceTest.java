package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.CompactFloatReference;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorTestStorage;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class MetalCastReferenceTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void every16BitValueDecodesAgainstAnIndependentReference(DType dtype) {
        short[] bits = new short[65_536];
        for (int index = 0; index < bits.length; index++) { bits[index] = (short) index; }
        Tensor source = TensorTestStorage.from16Bits(bits, dtype, 256, 256);
        Tensor output = source.to(DType.FLOAT32);
        assertFalse(TensorTestStorage.hasHostStorage(output));
        float[] actual = output.toFloatArray();
        for (int index = 0; index < bits.length; index++) {
            float expected = dtype == DType.FLOAT16 ? CompactFloatReference.half(index)
                    : Float.intBitsToFloat(index << 16);
            assertSameValueBits(expected, actual[index]);
        }
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void randomFloat32CastsMatchRneIncludingSubnormalAndBoundaryPatterns(DType dtype) {
        float[] values = new float[100_016];
        int[] special = {0, 0x80000000, 1, 0x007fffff, 0x00800000, 0x33000000, 0x33800000, 0x387fc000,
                0x38800000, 0x477fe000, 0x477ff000, 0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc01234, 0xff801234};
        Random random = new Random(2026);
        for (int index = 0; index < values.length; index++) {
            values[index] = Float.intBitsToFloat(index < special.length ? special[index] : random.nextInt());
        }
        assertQuantized(values, dtype);
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void positiveAndNegativeHalfwayValuesRoundToEvenAndAdjacentFloatsRoundAway(DType dtype) {
        int limit = dtype == DType.FLOAT16 ? 0x7bff : 0x7f7f;
        float[] values = new float[limit * 6];
        for (int bits = 0; bits < limit; bits++) {
            float lower = decoded(bits, dtype), upper = decoded(bits + 1, dtype);
            float midpoint = (float) (((double) lower + upper) * 0.5);
            int base = bits * 6;
            values[base] = midpoint;
            values[base + 1] = Math.nextDown(midpoint);
            values[base + 2] = Math.nextUp(midpoint);
            values[base + 3] = -midpoint;
            values[base + 4] = -values[base + 1];
            values[base + 5] = -values[base + 2];
        }
        assertQuantized(values, dtype);
    }

    private void assertQuantized(float[] values, DType dtype) {
        Tensor source = Tensor.from(values);
        Tensor encoded = source.to(dtype);
        short[] actual = TensorTestStorage.bits16(encoded);
        float[] decoded = encoded.to(DType.FLOAT32).toFloatArray();
        for (int index = 0; index < values.length; index++) {
            short expected = dtype == DType.FLOAT16 ? Float.floatToFloat16(values[index])
                    : CompactFloatReference.bfloat(values[index]);
            if (!Float.isNaN(values[index])) assertEquals(expected, actual[index], "encoded index " + index);
            assertSameValueBits(decoded(Short.toUnsignedInt(expected), dtype), decoded[index]);
        }
        assertEquals((long) values.length * 2, encoded.storageBytes());
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void compactToCompactCastsUseBothFormatsRatherThanReinterpretingTheirBits(DType dtype) {
        DType target = dtype == DType.FLOAT16 ? DType.BFLOAT16 : DType.FLOAT16;
        short[] bits = new short[65_536];
        for (int index = 0; index < bits.length; index++) { bits[index] = (short) index; }
        Tensor source = TensorTestStorage.from16Bits(bits, dtype, 256, 256);
        Tensor encoded = source.permute(1, 0).to(target);
        float[] actual = encoded.to(DType.FLOAT32).toFloatArray();
        for (int index = 0; index < bits.length; index++) {
            int original = (index % 256) * 256 + index / 256;
            float value = decoded(original, dtype);
            int expected = Short.toUnsignedInt(target == DType.FLOAT16 ? Float.floatToFloat16(value)
                    : CompactFloatReference.bfloat(value));
            assertSameValueBits(decoded(expected, target), actual[index]);
        }
    }

    private static float decoded(int bits, DType dtype) {
        return dtype == DType.FLOAT16 ? CompactFloatReference.half(bits) : Float.intBitsToFloat(bits << 16);
    }

    private static void assertSameValueBits(float expected, float actual) {
        if (Float.isNaN(expected)) assertTrue(Float.isNaN(actual));
        else assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(actual));
    }
}
