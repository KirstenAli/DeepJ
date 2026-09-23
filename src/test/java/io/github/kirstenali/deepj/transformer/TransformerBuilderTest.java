package io.github.kirstenali.deepj.transformer;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Random;

public class TransformerBuilderTest {

    @Test
    void builder_requiresAllHyperparams() {
        Assertions.assertThrows(IllegalArgumentException.class, new DeepJOriginTransformerBuilder()::build);

        TransformerStack s = new DeepJOriginTransformerBuilder()
                .dModel(8).nHeads(2).dFF(16).nLayers(1).seed(1L)
                .build();
        Assertions.assertNotNull(s);
    }

    @Test
    void orbitBuilderForwardBackwardShapes() {
        TransformerStack stack = new DeepJOrbitTransformerBuilder()
                .dModel(8).nHeads(2).dFF(16).nLayers(2)
                .maxSeqLen(16)
                .seed(1L)
                .build();

        Tensor x = Tensor.random(4, 8, new Random(2));
        Tensor y = stack.forward(x);
        TestSupport.assertTensorShape(y, 4, 8);
        TestSupport.assertTensorShape(stack.backward(Tensor.ones(4, 8)), 4, 8);
    }

    @Test
    void orbitBuilderRequiresMaxSeqLen() {
        Assertions.assertThrows(IllegalArgumentException.class,
                () -> new DeepJOrbitTransformerBuilder()
                        .dModel(8).nHeads(2).dFF(16).nLayers(1)
                        .build());
    }

    @Test
    void prismBuilderForwardBackwardShapes() {
        TransformerStack stack = new DeepJPrismTransformerBuilder()
                .dModel(8).nHeads(2).dFF(16).nLayers(2)
                .maxSeqLen(16).qRank(4).kvRank(2)
                .seed(1L)
                .build();

        Tensor x = Tensor.random(4, 8, new Random(3));
        Tensor y = stack.forward(x);
        TestSupport.assertTensorShape(y, 4, 8);
        TestSupport.assertTensorShape(stack.backward(Tensor.ones(4, 8)), 4, 8);
    }

    @Test
    void prismBuilderRequiresRanks() {
        Assertions.assertThrows(IllegalArgumentException.class,
                () -> new DeepJPrismTransformerBuilder()
                        .dModel(8).nHeads(2).dFF(16).nLayers(1)
                        .maxSeqLen(16)
                        .build());
    }
}
