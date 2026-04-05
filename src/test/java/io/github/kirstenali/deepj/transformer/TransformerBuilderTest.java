package io.github.kirstenali.deepj.transformer;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.transformer.TransformerBuilder.BlockType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Random;

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

    @Test
    void llamaStack_forwardBackward_shapes() {
        TransformerStack stack = new TransformerBuilder()
                .blockType(BlockType.LLAMA)
                .dModel(8).nHeads(2).dFF(16).nLayers(2)
                .maxSeqLen(16)
                .seed(1L)
                .build();

        Tensor x = Tensor.random(4, 8, new Random(2));
        Tensor y = stack.forward(x);
        TestSupport.assertTensorShape(y, 4, 8);

        Tensor gx = stack.backward(Tensor.ones(4, 8));
        TestSupport.assertTensorShape(gx, 4, 8);
    }

    @Test
    void llamaStack_requiresMaxSeqLen() {
        Assertions.assertThrows(IllegalArgumentException.class, () ->
                new TransformerBuilder()
                        .blockType(BlockType.LLAMA)
                        .dModel(8).nHeads(2).dFF(16).nLayers(1)
                        .build());
    }

    @Test
    void deepSeekStack_forwardBackward_shapes() {
        TransformerStack stack = new TransformerBuilder()
                .blockType(BlockType.DEEPSEEK)
                .dModel(8).nHeads(2).dFF(16).nLayers(2)
                .maxSeqLen(16).qRank(4).kvRank(2)
                .seed(1L)
                .build();

        Tensor x = Tensor.random(4, 8, new Random(3));
        Tensor y = stack.forward(x);
        TestSupport.assertTensorShape(y, 4, 8);

        Tensor gx = stack.backward(Tensor.ones(4, 8));
        TestSupport.assertTensorShape(gx, 4, 8);
    }

    @Test
    void deepSeekStack_requiresRanks() {
        Assertions.assertThrows(IllegalArgumentException.class, () ->
                new TransformerBuilder()
                        .blockType(BlockType.DEEPSEEK)
                        .dModel(8).nHeads(2).dFF(16).nLayers(1)
                        .maxSeqLen(16)
                        // missing qRank and kvRank
                        .build());
    }
}
