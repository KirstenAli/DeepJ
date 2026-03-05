package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.activations.ReLU;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.optimisers.Parameter;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class FNNTest {

    @Test
    void forward_backward_shapesAndGradients() {
        FNN mlp = new FNN(3, new int[]{4}, 2, ReLU::new, new Random(123));

        Tensor x = TestSupport.tensor(new double[][]{
                {1.0, 0.0, -1.0},
                {0.5, 2.0,  1.0}
        });

        Tensor y = mlp.forward(x);
        TestSupport.assertTensorShape(y, 2, 2);

        Tensor gradOut = Tensor.ones(y.rows, y.cols);
        Tensor gradIn = mlp.backward(gradOut);
        TestSupport.assertTensorShape(gradIn, 2, 3);

        double totalGrad = 0.0;
        for (Parameter p : mlp.parameters()) totalGrad += TestSupport.tensorSumAbs(p.grad);
        assertTrue(totalGrad > 0.0, "Expected some parameter gradients");
    }

    @Test
    void constructor_validation() {
        assertThrows(IllegalArgumentException.class, () -> new FNN(0, new int[]{4}, 2, ReLU::new, new Random(1)));
        assertThrows(IllegalArgumentException.class, () -> new FNN(3, null, 2, ReLU::new, new Random(1)));
        assertThrows(IllegalArgumentException.class, () -> new FNN(3, new int[]{4}, 0, ReLU::new, new Random(1)));
        assertThrows(IllegalArgumentException.class, () -> new FNN(3, new int[]{4}, 2, null, new Random(1)));
        assertThrows(IllegalArgumentException.class, () -> new FNN(3, new int[]{4}, 2, ReLU::new, null));
    }
}
