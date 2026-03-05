package io.github.kirstenali.deepj.transformer.embeddings;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class EmbeddingTest {

    @Test
    void forward_pullsCorrectRows_fromWeightMatrix() {
        Embedding emb = new Embedding(5, 3, new Random(1));

        // Overwrite weights with deterministic values: row i = [i, i+1, i+2]
        for (int i = 0; i < 5; i++) {
            emb.weight().value.data[i][0] = i;
            emb.weight().value.data[i][1] = i + 1;
            emb.weight().value.data[i][2] = i + 2;
        }

        Tensor out = emb.forward(new int[]{3, 1, 3});
        TestSupport.assertTensorShape(out, 3, 3);

        assertArrayEquals(new double[]{3, 4, 5}, out.data[0], 1e-12);
        assertArrayEquals(new double[]{1, 2, 3}, out.data[1], 1e-12);
        assertArrayEquals(new double[]{3, 4, 5}, out.data[2], 1e-12);
    }

    @Test
    void backward_accumulatesGradients_forRepeatedTokenIds() {
        Embedding emb = new Embedding(4, 2, new Random(2));
        emb.weight().zeroGrad();

        emb.forward(new int[]{1, 1, 3});

        Tensor gradOut = TestSupport.tensor(new double[][]{
                {1.0, 2.0},
                {3.0, 4.0},
                {5.0, 6.0}
        });
        emb.backward(gradOut);

        // id=1 appears twice => grads sum
        assertEquals(1.0 + 3.0, emb.weight().grad.data[1][0], 1e-12);
        assertEquals(2.0 + 4.0, emb.weight().grad.data[1][1], 1e-12);

        // id=3 appears once
        assertEquals(5.0, emb.weight().grad.data[3][0], 1e-12);
        assertEquals(6.0, emb.weight().grad.data[3][1], 1e-12);
    }

    @Test
    void forward_rejectsOutOfRangeIds() {
        Embedding emb = new Embedding(3, 2, new Random(3));
        assertThrows(IllegalArgumentException.class, () -> emb.forward(new int[]{-1}));
        assertThrows(IllegalArgumentException.class, () -> emb.forward(new int[]{3}));
    }
}
