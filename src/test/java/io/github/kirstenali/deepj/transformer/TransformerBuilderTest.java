
package io.github.kirstenali.deepj.transformer;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class TransformerBuilderTest {

    @Test
    void builder_requiresAllHyperparams() {
        TransformerBuilder b = new TransformerBuilder();
        // Builder validates required hyperparameters at build time.
        Assertions.assertThrows(IllegalArgumentException.class, b::build);

        TransformerStack s = new TransformerBuilder()
                .dModel(8).nHeads(2).dFF(16).nLayers(1).seed(1L)
                .build();
        Assertions.assertNotNull(s);
    }
}
