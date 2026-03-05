
package io.github.kirstenali.deepj.loss;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class LossTest {

    @Test
    void mseLoss_matchesSimpleCase() {
        MSELoss mse = new MSELoss();
        Tensor yHat = TestSupport.tensor(new double[][]{{1, 2}});
        Tensor y = TestSupport.tensor(new double[][]{{3, 0}});

        double loss = mse.loss(yHat, y);
        // mean((1-3)^2, (2-0)^2) = mean(4,4) = 4
        Assertions.assertEquals(4.0, loss, 1e-12);

        Tensor g = mse.gradient(yHat, y);
        // d/dyHat mean((yHat-y)^2) = 2*(yHat-y)/N ; N=2
        TestSupport.assertTensorAllClose(g, TestSupport.tensor(new double[][]{{-2, 2}}), 1e-12);
    }

    @Test
    void crossEntropyLoss_decreasesWhenCorrectLogitIncreases() {
        // 1 token, vocab 3
        Tensor logits1 = TestSupport.tensor(new double[][]{{0, 0, 0}});
        Tensor logits2 = TestSupport.tensor(new double[][]{{0, 0, 5}});
        int[] target = new int[]{2};

        double l1 = CrossEntropyLoss.loss(logits1, target);
        double l2 = CrossEntropyLoss.loss(logits2, target);
        Assertions.assertTrue(l2 < l1, "loss should be lower when correct class logit is higher");
    }

    @Test
    void crossEntropyGradient_shapeAndRowSumZero() {
        Tensor logits = TestSupport.tensor(new double[][]{
                {1, 2, 3},
                {3, 2, 1}
        });
        int[] target = new int[]{2, 0};

        Tensor g = CrossEntropyLoss.gradient(logits, target);
        TestSupport.assertTensorShape(g, 2, 3);

        for (int r = 0; r < g.rows; r++) {
            double sum = 0.0;
            for (int c = 0; c < g.cols; c++) sum += g.data[r][c];
            Assertions.assertEquals(0.0, sum, 1e-9, "softmax-crossentropy grad rows should sum to 0");
        }
    }
}
