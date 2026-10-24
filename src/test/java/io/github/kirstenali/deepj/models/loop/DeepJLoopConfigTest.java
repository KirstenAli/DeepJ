package io.github.kirstenali.deepj.models.loop;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.assertThrows;

class DeepJLoopConfigTest {

    @Test
    void invalidRecurrenceCountsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> counts(-1, 1, 1, 2));
        assertThrows(IllegalArgumentException.class, () -> counts(1, 0, 1, 2));
        assertThrows(IllegalArgumentException.class, () -> counts(1, 1, -1, 2));
        assertThrows(IllegalArgumentException.class, () -> counts(1, 1, 1, 0));
        assertThrows(ArithmeticException.class, () -> counts(1, Integer.MAX_VALUE, 1, 2));
        assertThrows(ArithmeticException.class, () -> counts(1, 2, 1, Integer.MAX_VALUE));
    }

    @ParameterizedTest
    @ValueSource(floats = {0.0f, -0.5f, Float.NaN, Float.POSITIVE_INFINITY})
    void invalidResidualScalesAreRejected(float scale) {
        assertThrows(IllegalArgumentException.class, () ->
                new DeepJLoopConfig(7, 3, 8, 4, 2, 1, 1, 1, 2, 12, true, 0.2f, 1.0f, scale));
        assertThrows(IllegalArgumentException.class, () ->
                new io.github.kirstenali.deepj.layers.transformer.blocks.DeepJNovaTransformerBlock(
                        counts(1, 1, 1, 2).backbone(), new java.util.Random(1L), scale));
    }

    @Test
    void backboneValidationStillApplies() {
        assertThrows(IllegalArgumentException.class, () ->
                new DeepJLoopConfig(7, 3, 8, 4, 3, 1, 1, 1, 2, 12));
        assertThrows(IllegalArgumentException.class, () ->
                new DeepJLoopConfig(7, 3, 12, 4, 2, 1, 1, 1, 2, 12));
    }

    private static DeepJLoopConfig counts(int prelude, int core, int coda, int passes) {
        return new DeepJLoopConfig(7, 3, 8, 4, 2, prelude, core, coda, passes, 12);
    }
}
