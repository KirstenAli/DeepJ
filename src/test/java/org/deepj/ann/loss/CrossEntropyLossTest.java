package org.deepj.ann.loss;

import org.deepj.ann.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class CrossEntropyLossTest {

    @Test
    public void lossMatchesSimpleCase() {
        // 2 examples, vocab=3
        Tensor logits = new Tensor(new double[][]{
                {Math.log(0.1), Math.log(0.2), Math.log(0.7)},
                {Math.log(0.8), Math.log(0.1), Math.log(0.1)}
        });
        int[] y = new int[]{2, 0};

        double loss = CrossEntropyLoss.loss(logits, y);

        double expected = (-Math.log(0.7) - Math.log(0.8)) / 2.0;
        assertEquals(expected, loss, 1e-6);
    }

    @Test
    public void gradientRowsSumToZero() {
        Tensor logits = new Tensor(new double[][]{
                {1.0, 2.0, 3.0},
                {3.0, 2.0, 1.0}
        });
        int[] y = new int[]{2, 0};
        Tensor grad = CrossEntropyLoss.gradient(logits, y);

        for (int r = 0; r < grad.rows; r++) {
            double sum = 0.0;
            for (int c = 0; c < grad.cols; c++) sum += grad.data[r][c];
            assertEquals(0.0, sum, 1e-9);
        }
    }
}
