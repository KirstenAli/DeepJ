package io.github.kirstenali.deepj.transformer;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Random;

public class TransformerBuilderTest {

    @Test
    void builder_requiresAllHyperparams() {
        Assertions.assertThrows(IllegalArgumentException.class, new GPTTransformerBuilder()::build);

        TransformerStack s = new GPTTransformerBuilder()
                .dModel(8).nHeads(2).dFF(16).nLayers(1).seed(1L)
                .build();
        Assertions.assertNotNull(s);
    }

    @Test
    void llamaBuilder_forwardBackward_shapes() {
        TransformerStack stack = new LlamaTransformerBuilder()
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
    void llamaBuilder_requiresMaxSeqLen() {
        Assertions.assertThrows(IllegalArgumentException.class,
                () -> new LlamaTransformerBuilder()
                        .dModel(8).nHeads(2).dFF(16).nLayers(1)
                        .build());
    }

    @Test
    void deepSeekBuilder_forwardBackward_shapes() {
        TransformerStack stack = new DeepSeekTransformerBuilder()
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
    void deepSeekBuilder_requiresRanks() {
        Assertions.assertThrows(IllegalArgumentException.class,
                () -> new DeepSeekTransformerBuilder()
                        .dModel(8).nHeads(2).dFF(16).nLayers(1)
                        .maxSeqLen(16)
                        .build());
    }
}
