package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import static org.junit.jupiter.api.Assertions.*;

class CpuNativeAdamWTest {
    private record State(Tensor w, Tensor g, Tensor m, Tensor v) {}

    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @CsvSource({"255,0", "256,0.01", "257,0", "65539,0.01"})
    void tenUpdatesAndOffsetTailsMatchOriginalFloatArithmetic(int count, float decay) {
        State actual = state(count);
        State expected = copy(actual);
        for (int step = 0; step < 10; step++) {
            reference(expected, decay);
            Tensor.backend().adamWUpdate(actual.w, actual.g, actual.m, actual.v,
                    .0001f, .9f, .999f, 1e-8f, decay, .91f, .003f);
        }
        assertArrayEquals(expected.w.toFloatArray(), actual.w.toFloatArray());
        assertArrayEquals(expected.m.toFloatArray(), actual.m.toFloatArray());
        assertArrayEquals(expected.v.toFloatArray(), actual.v.toFloatArray());
        assertEquals(-17, actual.w.memory().get(0));
        assertEquals(-17, actual.w.memory().get(count + 1));
    }

    @Test void stridedStatesKeepTheLogicalFallbackAndSharedBuffersAreRejected() {
        State expected = copy(state(512));
        State actual = new State(new Tensor(expected.w).reshape(16, 32).transpose(0, 1),
                new Tensor(expected.g).reshape(16, 32).transpose(0, 1),
                new Tensor(expected.m).reshape(16, 32).transpose(0, 1),
                new Tensor(expected.v).reshape(16, 32).transpose(0, 1));
        reference(expected, .01f);
        Tensor.backend().adamWUpdate(actual.w, actual.g, actual.m, actual.v,
                .0001f, .9f, .999f, 1e-8f, .01f, .91f, .003f);
        assertArrayEquals(expected.w.toFloatArray(), actual.w.transpose(0, 1).toFloatArray());
        assertThrows(IllegalArgumentException.class, () -> Tensor.backend().adamWUpdate(actual.w,
                actual.w, actual.m, actual.v, .0001f, .9f, .999f, 1e-8f, .01f, .91f, .003f));
    }

    private static void reference(State state, float decay) {
        float[] w = state.w.rawData(), g = state.g.rawData(), m = state.m.rawData(), v = state.v.rawData();
        for (int i = 0; i < w.length; i++) {
            float first = .9f * m[i] + (1 - .9f) * g[i];
            float second = .999f * v[i] + (1 - .999f) * (g[i] * g[i]);
            m[i] = first;
            v[i] = second;
            float update = (.0001f * (first / .91f)) / ((float) Math.sqrt(second / .003f) + 1e-8f);
            if (decay != 0) update += .0001f * decay * w[i];
            w[i] -= update;
        }
    }

    private static State state(int count) {
        return new State(values(count, 1), values(count, 3), values(count, .01f), values(count, .01f));
    }

    private static State copy(State source) {
        return new State(new Tensor(source.w), new Tensor(source.g), new Tensor(source.m), new Tensor(source.v));
    }

    private static Tensor values(int count, float scale) {
        var random = new Random(count);
        float[] values = new float[count + 2];
        values[0] = -17;
        values[count + 1] = -17;
        for (int index = 1; index <= count; index++) { values[index] = random.nextFloat() * scale; }
        return Tensor.from(values, count + 2).slice(0, 1, count + 1);
    }
}
