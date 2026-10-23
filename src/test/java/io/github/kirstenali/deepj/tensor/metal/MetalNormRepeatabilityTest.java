package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MetalNormRepeatabilityTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @ValueSource(ints = {17, 65_537, 1_048_576})
    void identicalGradientsProduceIdenticalClippingNorms(int count) {
        List<Tensor> gradients = gradients(count);
        float expected = metal.l2Norm(gradients);
        assertEquals(referenceNorm(gradients), expected, Math.max(1e-6, expected * 2e-6));
        for (int round = 0; round < 64; round++) {
            assertEquals(Float.floatToRawIntBits(expected),
                    Float.floatToRawIntBits(metal.l2Norm(gradients)), "round " + round);
            metal.releaseTemporaryResources();
        }
    }

    private static List<Tensor> gradients(int count) {
        Random random = new Random(2026);
        List<Tensor> result = new ArrayList<>();
        for (float scale : new float[]{0.0001f, 0.01f, 1.0f, 10.0f, 100.0f}) {
            Tensor tensor = new Tensor(1, count).retainDeviceBuffer();
            for (int index = 0; index < count; index++) {
                tensor.data[index] = (random.nextFloat() - 0.5f) * scale;
            }
            result.add(tensor);
        }
        return result;
    }

    private static double referenceNorm(List<Tensor> gradients) {
        double squares = 0;
        for (Tensor gradient : gradients) {
            for (float value : gradient.data) {
                squares += (double) value * value;
            }
        }
        return Math.sqrt(squares);
    }
}
