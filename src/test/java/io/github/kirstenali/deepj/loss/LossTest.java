package io.github.kirstenali.deepj.loss;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class LossTest {

    @Test
    void mseLoss_matchesSimpleCase() {
        MSELoss mse = new MSELoss();
        Tensor yHat = Tensor.from2D(new float[][]{{1, 2}});
        Tensor y = Tensor.from2D(new float[][]{{3, 0}});

        float loss = mse.loss(yHat, y);
        // mean((1-3)^2, (2-0)^2) = mean(4,4) = 4
        Assertions.assertEquals(4.0f, loss, 1e-12f);

        Tensor g = mse.gradient(yHat, y);
        // d/dyHat mean((yHat-y)^2) = 2*(yHat-y)/N ; N=2
        TestSupport.assertTensorAllClose(g, Tensor.from2D(new float[][]{{-2, 2}}), 1e-12f);
    }

    @Test
    void crossEntropyLoss_decreasesWhenCorrectLogitIncreases() {
        // 1 token, vocab 3
        Tensor logits1 = Tensor.from2D(new float[][]{{0, 0, 0}});
        Tensor logits2 = Tensor.from2D(new float[][]{{0, 0, 5}});
        int[] target = new int[]{2};

        float l1 = CrossEntropyLoss.loss(logits1, target);
        float l2 = CrossEntropyLoss.loss(logits2, target);
        Assertions.assertTrue(l2 < l1, "loss should be lower when correct class logit is higher");
    }

    @Test
    void crossEntropyGradient_shapeAndRowSumZero() {
        Tensor logits = Tensor.from2D(new float[][]{
                {1, 2, 3},
                {3, 2, 1}
        });
        int[] target = new int[]{2, 0};

        Tensor g = CrossEntropyLoss.gradient(logits, target);
        TestSupport.assertTensorShape(g, 2, 3);

        for (int r = 0; r < g.rows; r++) {
            float sum = 0.0f;
            for (int c = 0; c < g.cols; c++) sum += g.data[r * g.cols + c];
            Assertions.assertEquals(0.0f, sum, 1e-6f, "softmax-crossentropy grad rows should sum to 0");
        }
    }

    @Test
    void crossEntropyGradient_matchesSoftmaxMinusOneHot_singleRow() {
        Tensor logits = Tensor.from2D(new float[][]{{1, 2, 3}});
        int[] target = new int[]{2};

        Tensor g = CrossEntropyLoss.gradient(logits, target);
        TestSupport.assertTensorShape(g, 1, 3);

        // softmax(logits)
        double a = Math.exp(1);
        double b = Math.exp(2);
        double c = Math.exp(3);
        double s = a + b + c;

        double p0 = a / s;
        double p1 = b / s;
        double p2 = c / s;

        // grad = softmax - oneHot(target)
        Assertions.assertEquals(p0, g.data[0], 1e-6f);
        Assertions.assertEquals(p1, g.data[1], 1e-6f);
        Assertions.assertEquals(p2 - 1.0f, g.data[2], 1e-6f);
    }

    @Test
    void crossEntropyRejectsNonIntegerTensorTargets() {
        Tensor logits = Tensor.from2D(new float[][]{{1, 2, 3}});
        Tensor badTargets = Tensor.from2D(new float[][]{{1.5f}});

        Assertions.assertThrows(IllegalArgumentException.class,
                () -> new CrossEntropyLoss().loss(logits, badTargets));
    }
}
