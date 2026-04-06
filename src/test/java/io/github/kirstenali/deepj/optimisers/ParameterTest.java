
package io.github.kirstenali.deepj.optimisers;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class ParameterTest {

    @Test
    void zeroGrad_clearsToZeros() {
        Parameter p = new Parameter(new Tensor(new double[][]{{1,2},{3,4}}));
        Tensor originalGrad = p.grad;

        p.grad.data[0][0] = 7.0;
        p.grad.data[1][1] = -3.0;
        p.zeroGrad();

        Assertions.assertSame(originalGrad, p.grad);
        Assertions.assertEquals(0.0, p.grad.data[0][0], 1e-12);
        Assertions.assertEquals(0.0, p.grad.data[1][1], 1e-12);
        Assertions.assertEquals(2, p.grad.rows);
        Assertions.assertEquals(2, p.grad.cols);
    }

    @Test
    void zeroGrad_gpuTaggedGrad_replacesBuffer() {
        Parameter p = new Parameter(new Tensor(new double[][]{{1, 2}, {3, 4}}));
        Tensor originalGrad = p.grad;
        p.grad.setGpuTag(new Object());

        p.zeroGrad();

        Assertions.assertNotSame(originalGrad, p.grad);
        Assertions.assertEquals(0.0, p.grad.data[0][0], 1e-12);
        Assertions.assertEquals(0.0, p.grad.data[1][1], 1e-12);
    }
}
