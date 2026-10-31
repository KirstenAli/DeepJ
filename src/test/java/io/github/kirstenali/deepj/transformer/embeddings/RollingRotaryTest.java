package io.github.kirstenali.deepj.transformer.embeddings;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class RollingRotaryTest {
    @BeforeEach
    void useCpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @ValueSource(ints = {0, 3, 7, 8, 31, 1024, 16383})
    void onDemandRowsMatchIndependentLargerPositionTablesExactly(int offset) {
        var shortTable = new RotaryEmbedding(8, 8);
        var longTable = new RotaryEmbedding(8, offset + 9);
        Tensor input = Tensor.random(6, 8, new Random(53));
        try (var ignored = GradMode.noGrad()) {
            assertArrayEquals(longTable.apply(input, 2, 3, offset).rawData(),
                    shortTable.applyInference(input, 2, 3, offset).rawData());
        }
    }

    @Test
    void extendedPositionsRejectTrainingInvalidLayoutsAndIntegerOverflow() {
        var rotary = new RotaryEmbedding(4, 4);
        Tensor row = Tensor.ones(1, 4);
        assertThrows(IllegalStateException.class, () -> rotary.applyInference(row, 1, 1, 4));
        try (var ignored = GradMode.noGrad()) {
            assertThrows(IllegalArgumentException.class, () -> rotary.applyInference(row, 1, 1, -1));
            assertThrows(IllegalArgumentException.class, () -> rotary.applyInference(row, 2, 1, 4));
            assertThrows(ArithmeticException.class, () -> rotary.applyInference(row, 1, 1, Integer.MAX_VALUE));
        }
    }
}
