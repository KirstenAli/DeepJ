package io.github.kirstenali.deepj.transformer;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Random;

public class TransformerStackTest {

    @Test
    void forwardBackward_shapes() {
        TransformerStack stack = new TransformerBuilder()
                .dModel(8)
                .nHeads(2)
                .dFF(16)
                .nLayers(2)
                .seed(1L)
                .build();

        // 4 tokens x dModel (uniform RNG is fine for shape/grad tests)
        Tensor x = Tensor.random(4, 8, new Random(2));
        Tensor y = stack.forward(x);
        TestSupport.assertTensorShape(y, 4, 8);

        Tensor gradOut = Tensor.ones(4, 8);
        Tensor gx = stack.backward(gradOut);
        TestSupport.assertTensorShape(gx, 4, 8);

        org.junit.jupiter.api.Assertions.assertFalse(stack.parameters().isEmpty());
    }
}
