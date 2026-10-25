package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.AttentionOps;
import io.github.kirstenali.deepj.tensor.CausalAttentionResult;
import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class MetalAttentionValidationTest extends MetalDifferentialTestSupport {
    @BeforeEach
    void requireTiledDevice() { assumeTrue(MetalNative.TILED_ATTENTION_AVAILABLE); }

    @ParameterizedTest
    @CsvSource({"31,64,1", "33,64,1", "32,32,1", "32,64,0", "32,64,3"})
    void invalidShapesCannotEnterTheTiledKernel(int rows, int columns, int heads) {
        Tensor input = Tensor.ones(rows, columns);
        assertFalse(metal.supportsCausalAttention(input, input, input, heads, 0.125f));
        assertThrows(IllegalArgumentException.class,
                () -> metal.causalAttention(input, input, input, heads, 0.125f));
    }

    @ParameterizedTest
    @ValueSource(floats = {Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY})
    void nonfiniteScalesAreNotPassedToTheBlockSoftmax(float scale) {
        Tensor input = Tensor.ones(256, 64);
        assertFalse(metal.supportsCausalAttention(input, input, input, 1, scale));
        assertThrows(IllegalArgumentException.class,
                () -> metal.causalAttention(input, input, input, 1, scale));
    }

    @Test
    void shapeMismatchesAndInvalidSavedStateAreRejected() {
        Tensor input = Tensor.ones(256, 64), wrong = Tensor.ones(256, 32);
        assertFalse(metal.supportsCausalAttention(input, wrong, input, 1, 0.125f));
        assertFalse(metal.supportsCausalAttention(input, input, wrong, 1, 0.125f));
        assertThrows(IllegalArgumentException.class, () -> metal.causalAttention(input, wrong, input, 1, 0.125f));
        assertThrows(IllegalArgumentException.class, () -> metal.causalAttention(input, input, wrong, 1, 0.125f));
        var saved = metal.causalAttention(input, input, input, 1, 0.125f);
        assertInvalidBackward(input, saved, wrong);
        assertInvalidBackward(input, new CausalAttentionResult(wrong, saved.normalization()), input);
        assertInvalidBackward(input, new CausalAttentionResult(input, Tensor.ones(255, 2)), input);
        assertInvalidBackward(input, new CausalAttentionResult(input, Tensor.ones(256, 3)), input);
    }

    private void assertInvalidBackward(Tensor input, CausalAttentionResult saved, Tensor gradient) {
        assertThrows(IllegalArgumentException.class,
                () -> metal.causalAttentionBackward(input, input, input, saved, gradient, 1, 0.125f));
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 31, 32, 33, 255, 257})
    void shortAndRaggedSequencesRetainTheReferencePath(int length) {
        Tensor input = random(length, 64, 131);
        assertFalse(metal.supportsCausalAttention(input, input, input, 1, 0.125f));
        Tensor scores = metal.batchedMatmul(input, input, 1, false, true);
        Tensor probabilities = metal.causalSoftmax(scores, length, 0.125f);
        Tensor expected = metal.batchedMatmul(probabilities, input, 1, false, false);
        Tensor actual = AttentionOps.causal(input, input, input, 1, 0.125f);
        expected.materialize();
        actual.materialize();
        assertArrayEquals(expected.rawData(), actual.rawData());
    }

    @Test
    void nativeValidationRejectsMalformedTiledCommandsWithoutExecutingThem() {
        Tensor input = Tensor.ones(32, 64);
        int id = metal.graph.ensureGpuBuffer(input).id;
        metal.graph.flush();
        int[] command = {ComputeGraph.OP_TILED_CAUSAL_ATTENTION,
                id, id, id, id, id, 32, 1, 1, Float.floatToRawIntBits(0.125f)};
        assertThrows(RuntimeException.class, () -> MetalNative.nativeFlushOps(command, command.length));
        command[6] = 33;
        assertThrows(RuntimeException.class, () -> MetalNative.nativeFlushOps(command, command.length));
        command[6] = 32;
        command[9] = Float.floatToRawIntBits(Float.NaN);
        assertThrows(RuntimeException.class, () -> MetalNative.nativeFlushOps(command, command.length));
    }
}
