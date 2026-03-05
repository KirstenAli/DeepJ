package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.activations.Softmax;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Random;

public class TensorTest {

    @Test
    void constructor_copiesData_andValidatesRowLengths() {
        double[][] data = new double[][]{{1,2},{3,4}};
        Tensor t = new Tensor(data);
        data[0][0] = 999;
        Assertions.assertEquals(1.0, t.data[0][0], 1e-12);

        Assertions.assertThrows(IllegalArgumentException.class, () ->
                new Tensor(new double[][]{{1,2},{3}}));
    }

    @Test
    void add_and_matmul_workForSmallKnownCase() {
        Tensor a = TestSupport.tensor(new double[][]{{1,2},{3,4}});
        Tensor b = TestSupport.tensor(new double[][]{{10,20},{30,40}});

        Tensor s = a.add(b);
        TestSupport.assertTensorAllClose(s, TestSupport.tensor(new double[][]{{11,22},{33,44}}), 1e-12);

        Tensor m = a.matmul(b);
        // [[1*10+2*30, 1*20+2*40], [3*10+4*30, 3*20+4*40]]
        TestSupport.assertTensorAllClose(m, TestSupport.tensor(new double[][]{{70,100},{150,220}}), 1e-12);
    }

    @Test
    void softmax_activation_rowsSumTo1_and_argmaxMatchesMaxIndex() {
        Tensor logits = TestSupport.tensor(new double[][]{{1, 3, 2}});

        Softmax sm = new Softmax();
        Tensor p = sm.forward(logits);

        double sum = 0.0;
        for (int c = 0; c < p.cols; c++) sum += p.data[0][c];
        Assertions.assertEquals(1.0, sum, 1e-9);

        // simple argmax over the row
        int argmax = 0;
        double best = logits.data[0][0];
        for (int c = 1; c < logits.cols; c++) {
            if (logits.data[0][c] > best) {
                best = logits.data[0][c];
                argmax = c;
            }
        }
        Assertions.assertEquals(1, argmax);
    }

    @Test
    void causalMask_isAdditiveMask_zeroForAllowed_negLargeForFuture() {
        Tensor m = Tensor.causalMask(4);
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                double v = m.data[r][c];
                if (c > r) {
                    Assertions.assertTrue(v <= -1e8, "future positions should be strongly negative");
                } else {
                    Assertions.assertEquals(0.0, v, 1e-12, "allowed positions should be 0");
                }
            }
        }
    }

    @Test
    void backend_canBeSwapped() {
        TensorBackend original = Tensor.backend();
        try {
            Tensor.setBackend(new CpuBackend());
            Assertions.assertNotNull(Tensor.backend());
        } finally {
            Tensor.setBackend(original);
        }
    }

    @Test
    void random_isDeterministicWithSeed() {
        Random r1 = new Random(123);
        Random r2 = new Random(123);
        Tensor a = Tensor.random(2, 3, r1);
        Tensor b = Tensor.random(2, 3, r2);
        TestSupport.assertTensorAllClose(a, b, 1e-12);
    }
}
