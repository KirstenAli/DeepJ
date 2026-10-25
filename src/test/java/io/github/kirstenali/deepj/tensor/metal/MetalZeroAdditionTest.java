package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.persistence.ModelSerializer;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class MetalZeroAdditionTest extends MetalDifferentialTestSupport {
    @TempDir Path directory;

    @Test
    void optimizedAdditionMatchesBinaryAdditionBitForBit() {
        Tensor source = deviceValues();
        Tensor target = Tensor.zeros(source.rows, source.cols);
        Tensor expected = metal.add(new Tensor(target), source);
        assertTrue(MetalHostZeroInput.canSkip(target, source));
        metal.addInPlace(target, source);
        assertSameBits(expected, target);
    }

    @Test
    void nonzeroAndNegativeZeroTargetsUseTheExistingBinaryPath() {
        Tensor source = deviceValues();
        Tensor target = Tensor.zeros(source.rows, source.cols);
        target.rawData()[0] = -0.0f;
        target.rawData()[2] = 3;
        Tensor expected = metal.add(new Tensor(target), source);
        assertFalse(MetalHostZeroInput.canSkip(target, source));
        metal.addInPlace(target, source);
        assertSameBits(expected, target);
    }

    @Test
    void recordingCapturesValuesBeforeRawCpuArraysAreChanged() {
        Tensor input = Tensor.from2D(new float[][]{{2, 3, 4}});
        Tensor source = metal.multiplyScalar(input, 1);
        Tensor target = Tensor.zeros(1, 3);
        metal.addInPlace(target, source);
        input.rawData()[0] = 900;
        target.rawData()[0] = 800;
        assertSameBits(Tensor.from2D(new float[][]{{2, 3, 4}}), target);
    }

    @Test
    void retainedResultsAndTheirSourcesSurviveTemporaryCleanup() {
        Tensor source = metal.multiplyScalar(Tensor.ones(2, 3), 4).retainDeviceBuffer();
        Tensor target = Tensor.zeros(2, 3).retainDeviceBuffer();
        metal.addInPlace(target, source);
        metal.releaseTemporaryResources();
        assertSameBits(Tensor.from2D(new float[][]{{4, 4, 4}, {4, 4, 4}}), target);
        assertSameBits(target, source);
        assertTrue(target.retainsDeviceBuffer());
    }

    @Test
    void canonicalLeafGradientStillAccumulatesAndResets() {
        Parameter parameter = new Parameter(Tensor.ones(2, 3));
        Tensor canonical = parameter.grad();
        parameter.value.multiplyScalar(3).sumTensor().backward();
        assertSame(canonical, parameter.grad());
        assertSameBits(Tensor.from2D(new float[][]{{3, 3, 3}, {3, 3, 3}}), canonical);
        parameter.value.multiplyScalar(2).sumTensor().backward();
        assertSameBits(Tensor.from2D(new float[][]{{5, 5, 5}, {5, 5, 5}}), parameter.grad());
        parameter.zeroGrad();
        parameter.value.multiplyScalar(4).sumTensor().backward();
        assertSameBits(Tensor.from2D(new float[][]{{4, 4, 4}, {4, 4, 4}}), parameter.grad());
    }

    @Test
    void backendSwitchCanMaterializeTheResultFromItsOwningGraph() {
        Tensor source = metal.multiplyScalar(Tensor.ones(1, 3), 7);
        Tensor target = Tensor.zeros(1, 3);
        metal.addInPlace(target, source);
        Tensor.setBackend(cpu);
        assertArrayEquals(new float[]{7, 7, 7}, target.data());
        Tensor.setBackend(metal);
    }

    @Test
    void serializationMaterializesAndRestoresAnOptimizedResult() throws Exception {
        Tensor source = metal.multiplyScalar(Tensor.ones(1, 3), 7);
        Tensor target = Tensor.zeros(1, 3);
        metal.addInPlace(target, source);
        Path checkpoint = directory.resolve("float32.dj");
        ModelSerializer.save(List.of(new Parameter(target)), checkpoint);
        Parameter restored = new Parameter(Tensor.zeros(1, 3));
        ModelSerializer.load(List.of(restored), checkpoint);
        assertArrayEquals(new float[]{7, 7, 7}, restored.value.data());
    }

    private Tensor deviceValues() {
        return metal.multiplyScalar(Tensor.from2D(new float[][]{{0, -0.0f, 1, -1,
                Float.MIN_VALUE, -Float.MIN_VALUE, Float.MAX_VALUE, -Float.MAX_VALUE,
                Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN,
                Float.intBitsToFloat(0x7f800001)}}), 1);
    }

    private static void assertSameBits(Tensor expected, Tensor actual) {
        float[] before = expected.data(), after = actual.data();
        assertEquals(before.length, after.length);
        for (int index = 0; index < before.length; index++) {
            assertEquals(Float.floatToRawIntBits(before[index]), Float.floatToRawIntBits(after[index]),
                    "float bits at index " + index);
        }
    }
}
