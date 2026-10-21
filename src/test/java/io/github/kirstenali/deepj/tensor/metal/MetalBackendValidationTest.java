package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

class MetalBackendValidationTest extends MetalDifferentialTestSupport {

    @Test
    void broadcastVectorsMustMatchBothDimensions() {
        Tensor input = new Tensor(2, 4);
        assertThrows(IllegalArgumentException.class, () -> metal.addRowVector(input, new Tensor(2, 4)));
        assertThrows(IllegalArgumentException.class, () -> metal.addRowVector(input, new Tensor(1, 3)));
        assertThrows(IllegalArgumentException.class, () -> metal.addBroadcastCols(input, new Tensor(3, 1)));
        assertThrows(IllegalArgumentException.class, () -> metal.addBroadcastCols(input, new Tensor(2, 2)));
    }

    @Test
    void headsAndBatchesMustBePositiveAndDivideTheirInputs() {
        Tensor input = new Tensor(2, 4);
        assertThrows(IllegalArgumentException.class, () -> metal.splitHeads(input, 0));
        assertThrows(IllegalArgumentException.class, () -> metal.splitHeads(input, 3));
        assertThrows(IllegalArgumentException.class, () -> metal.mergeHeads(input, -1));
        assertThrows(IllegalArgumentException.class, () -> metal.mergeHeads(input, 3));
        assertThrows(IllegalArgumentException.class,
                () -> metal.batchedMatmul(input, input, 1, false, false));
    }

    @Test
    void gatherRejectsNegativeAndPastEndRows() {
        Tensor input = new Tensor(2, 4);
        assertThrows(IllegalArgumentException.class, () -> metal.sliceRows(input, new int[]{-1}));
        assertThrows(IllegalArgumentException.class, () -> metal.sliceRows(input, new int[]{2}));
    }

    @Test
    void causalScoresMustContainCompleteSquareHeads() {
        assertThrows(IllegalArgumentException.class, () -> metal.causalMask(new Tensor(2, 2), 0));
        assertThrows(IllegalArgumentException.class, () -> metal.causalMask(new Tensor(2, 2), 3));
        assertThrows(IllegalArgumentException.class, () -> metal.causalSoftmax(new Tensor(3, 2), 2, 1));
    }

    @Test
    void rotaryInputRequiresWholeSequencesAndPairedCoordinates() {
        Tensor table = new Tensor(2, 2);
        assertThrows(IllegalArgumentException.class,
                () -> metal.rotary(new Tensor(4, 4), table, table, 0, false));
        assertThrows(IllegalArgumentException.class,
                () -> metal.rotary(new Tensor(3, 4), table, table, 2, false));
        assertThrows(IllegalArgumentException.class,
                () -> metal.rotary(new Tensor(4, 5), table, table, 2, false));
    }

    @Test
    void rotaryTablesMustCoverTheSequenceAndMatchEachOther() {
        Tensor input = new Tensor(4, 4);
        Tensor table = new Tensor(2, 2);
        assertThrows(IllegalArgumentException.class,
                () -> metal.rotary(input, new Tensor(1, 2), table, 2, false));
        assertThrows(IllegalArgumentException.class,
                () -> metal.rotary(input, new Tensor(2, 1), table, 2, false));
        assertThrows(IllegalArgumentException.class,
                () -> metal.rotary(input, table, new Tensor(1, 2), 2, false));
        assertThrows(IllegalArgumentException.class,
                () -> metal.rotary(input, table, new Tensor(2, 1), 2, false));
    }

    @Test
    void normalizationRejectsMismatchedScalesAndWidths() {
        Tensor input = new Tensor(2, 4);
        Tensor gamma = new Tensor(1, 4);
        assertThrows(IllegalArgumentException.class, () -> metal.rmsNorm(input, new Tensor(2, 4), 1e-6f));
        assertThrows(IllegalArgumentException.class, () -> metal.rmsNorm(input, new Tensor(1, 3), 1e-6f));
        assertThrows(IllegalArgumentException.class,
                () -> metal.rmsNormBackward(input, input, new Tensor(3, 1), gamma));
        assertThrows(IllegalArgumentException.class,
                () -> metal.rmsNormBackward(input, input, new Tensor(2, 2), gamma));
        assertThrows(IllegalArgumentException.class,
                () -> metal.layerNormBackward(input, input, new Tensor(2, 1), 3));
        assertInvalidDeviations(input);
    }

    private void assertInvalidDeviations(Tensor input) {
        assertThrows(IllegalArgumentException.class,
                () -> metal.layerNormBackward(input, input, new Tensor(3, 1), 4));
        assertThrows(IllegalArgumentException.class,
                () -> metal.layerNormBackward(input, input, new Tensor(2, 2), 4));
    }

    @Test
    void scatterRejectsWrongIndexCountAndGradientWidth() {
        Tensor target = new Tensor(2, 4);
        assertThrows(IllegalArgumentException.class,
                () -> metal.scatterAddRows(target, new int[]{0}, new Tensor(2, 4)));
        assertThrows(IllegalArgumentException.class,
                () -> metal.scatterAddRows(target, new int[]{0}, new Tensor(1, 3)));
    }
}
