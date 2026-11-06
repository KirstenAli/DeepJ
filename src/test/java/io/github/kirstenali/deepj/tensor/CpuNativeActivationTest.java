package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class CpuNativeActivationTest {
    @BeforeEach void cpu() {
        Tensor.setBackend(new CpuBackend());
        assumeTrue(CpuBlas.AVAILABLE, "Optional Apple CPU library unavailable");
    }

    @Test void nativeOrdinaryValuesAndRoundingBoundaryFallbackAreBothExercised() {
        float[] gate = new float[1024], up = new float[1024], output = new float[1024];
        Arrays.fill(gate, 0.3f);
        Arrays.fill(up, -2.3f);
        assertTrue(CpuBlas.swiGlu(gate, 0, up, 0, output, 0, gate.length));
        assertEquation(gate, up, output);
        gate[17] = -0x1.0p-24f;
        assertFalse(CpuBlas.swiGlu(gate, 0, up, 0, output, 0, gate.length));
        Tensor actual = Tensor.backend().swiGlu(Tensor.from(gate, 1, 1, 1024), Tensor.from(up, 1, 1, 1024));
        assertEquation(gate, up, actual.toFloatArray());
    }

    @Test void sampledFiniteSubnormalSpecialAndBoundaryValuesPreserveJavaBits() {
        var random = new Random(20261106);
        for (int group = 0; group < 4; group++) { checkGroup(group, random); }
    }

    private static void checkGroup(int group, Random random) {
        int count = 262144;
        float[] a = new float[count + 31], b = new float[count + 31], padding = new float[count + 31];
        fillInputs(a, b, group, random);
        Arrays.fill(padding, 12345);
        Tensor parent = Tensor.from(padding), output = parent.slice(0, 19, count + 19).reshape(1, count);
        Tensor gate = Tensor.from(a).slice(0, 13, count + 13).reshape(1, count);
        Tensor up = Tensor.from(b).slice(0, 7, count + 7).reshape(1, count);
        CpuDenseSwiGlu.forward(gate, up, output);
        assertEquation(gate.toFloatArray(), up.toFloatArray(), output.toFloatArray());
        for (int i = 0; i < 19; i++) { assertEquals(12345, parent.get(i)); }
        for (int i = count + 19; i < padding.length; i++) { assertEquals(12345, parent.get(i)); }
    }

    private static void fillInputs(float[] a, float[] b, int group, Random random) {
        for (int i = 0; i < a.length; i++) {
            a[i] = gate(group, i, random);
            b[i] = group == 2 ? Float.intBitsToFloat(random.nextInt()) : random.nextFloat() * 8 - 4;
        }
    }

    @Test void aliasedDestinationsKeepTheIndependentJavaElementPath() {
        float[] a = new float[1024], b = new float[1024];
        Arrays.fill(a, 0.3f);
        Arrays.fill(b, -2.3f);
        Tensor gate = Tensor.from(a, 1, 1024), up = Tensor.from(b, 1, 1024);
        CpuDenseSwiGlu.forward(gate, up, gate);
        assertEquation(a, b, gate.toFloatArray());
        gate = Tensor.from(a, 1, 1024);
        CpuDenseSwiGlu.forward(gate, up, up);
        assertEquation(a, b, up.toFloatArray());
    }

    private static float gate(int group, int index, Random random) {
        if (group == 0) return random.nextFloat() * 160 - 80;
        if (group == 1) return finite(random);
        if (group == 2) return (random.nextFloat() * 2 - 1) * 0x1p-23f;
        float[] special = {Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, -100, -80, -1,
            -0f, 0f, 1, 80, 100, Float.MIN_VALUE, Float.MAX_VALUE, -Float.MAX_VALUE};
        return special[index % special.length];
    }

    private static float finite(Random random) {
        float value;
        do { value = Float.intBitsToFloat(random.nextInt()); }
        while (!Float.isFinite(value) || Math.abs(value) > 80);
        return value;
    }

    private static void assertEquation(float[] gate, float[] up, float[] actual) {
        assertEquals(gate.length, actual.length);
        for (int i = 0; i < gate.length; i++) {
            float sigmoid = 1f / (1f + (float) Math.exp(-gate[i]));
            float expected = (gate[i] * sigmoid) * up[i];
            assertEquals(Float.floatToIntBits(expected), Float.floatToIntBits(actual[i]), "element " + i);
        }
    }
}
