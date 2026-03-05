
package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

public class LinearTest {

    @Test
    void forwardBackward_shapes_and_basicGradSignals() {
        Linear lin = new Linear(2, 3, new Random(1));

        // overwrite weights for deterministic behavior
        List<Parameter> ps = lin.parameters();
        Parameter W = ps.get(0);
        Parameter b = ps.get(1);

        W.value = TestSupport.tensor(new double[][]{
                {1, 0, -1},
                {2, 1,  0}
        });
        b.value = TestSupport.tensor(new double[][]{{0.5, -0.5, 1.0}});

        Tensor x = TestSupport.tensor(new double[][]{
                {1, 2},
                {-1, 0}
        });

        Tensor y = lin.forward(x);
        TestSupport.assertTensorAllClose(y, TestSupport.tensor(new double[][]{
                // [1,2] * W = [1*1+2*2, 1*0+2*1, 1*-1+2*0] = [5,2,-1]; +b => [5.5,1.5,0]
                {5.5, 1.5, 0.0},
                // [-1,0] * W = [-1,0,1]; +b => [-0.5,-0.5,2]
                {-0.5, -0.5, 2.0}
        }), 1e-12);

        Tensor gradOut = TestSupport.tensor(new double[][]{
                {1, 1, 1},
                {2, 0, -1}
        });

        Tensor gx = lin.backward(gradOut);
        TestSupport.assertTensorShape(gx, 2, 2);

        // grads exist
        TestSupport.assertTensorShape(W.grad, 2, 3);
        TestSupport.assertTensorShape(b.grad, 1, 3);
    }
}
