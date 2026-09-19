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

        Assertions.assertEquals(4.0f, loss, 1e-12f);

        Tensor g = mse.gradient(yHat, y);

        TestSupport.assertTensorAllClose(g, Tensor.from2D(new float[][]{{-2, 2}}), 1e-12f);
    }

    @Test
    void crossEntropyLoss_decreasesWhenCorrectLogitIncreases() {

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

        double a = Math.exp(1);
        double b = Math.exp(2);
        double c = Math.exp(3);
        double s = a + b + c;

        double p0 = a / s;
        double p1 = b / s;
        double p2 = c / s;

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

    @Test
    void crossEntropyGradient_multiRow_isSoftmaxMinusOneHot_dividedByRowCount() {
        Tensor logits = Tensor.from2D(new float[][]{
                {1, 2, 3},
                {0, 0, 0}
        });
        int[] target = new int[]{2, 0};
        int n = logits.rows;

        Tensor g = CrossEntropyLoss.gradient(logits, target);
        TestSupport.assertTensorShape(g, 2, 3);

        double a = Math.exp(1), b = Math.exp(2), c = Math.exp(3), s = a + b + c;
        Assertions.assertEquals((a / s) / n, g.data[0], 1e-6f);
        Assertions.assertEquals((b / s) / n, g.data[1], 1e-6f);
        Assertions.assertEquals((c / s - 1.0) / n, g.data[2], 1e-6f);

        Assertions.assertEquals((1.0 / 3 - 1.0) / n, g.data[3], 1e-6f);
        Assertions.assertEquals((1.0 / 3) / n, g.data[4], 1e-6f);
        Assertions.assertEquals((1.0 / 3) / n, g.data[5], 1e-6f);
    }

    @Test
    void mseLoss_averagesOverAllElements_multiRow() {
        MSELoss mse = new MSELoss();
        Tensor yHat = Tensor.from2D(new float[][]{
                {1, 2},
                {3, 4}
        });
        Tensor y = Tensor.zeros(2, 2);

        Assertions.assertEquals(7.5f, mse.loss(yHat, y), 1e-6f);

        Tensor g = mse.gradient(yHat, y);
        TestSupport.assertTensorAllClose(g, Tensor.from2D(new float[][]{
                {0.5f, 1.0f},
                {1.5f, 2.0f}
        }), 1e-6f);
    }
}
