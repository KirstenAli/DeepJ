package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class GraphOpHistogramTest {
    @Test
    void opcodeNamesCoverEveryDeclaredOperation() {
        var names = GraphOpHistogram.opcodeNames();
        assertEquals("OP_ADD", names.get(GraphOpMetadata.OP_ADD));
        assertEquals("OP_LAYOUT_MATMUL", names.get(GraphOpMetadata.OP_LAYOUT_MATMUL));
        assertTrue(names.size() > 50);
    }

    @Test
    void recordingIsANoOpUnlessEnabled() {
        GraphOpHistogram.reset();
        GraphOpHistogram.record(new int[]{GraphOpMetadata.OP_ADD, 0, 0, 0}, 4);
        assertTrue(GraphOpHistogram.snapshot().isEmpty());
    }
}

