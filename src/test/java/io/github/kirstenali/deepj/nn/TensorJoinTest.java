package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import static io.github.kirstenali.deepj.nn.NeuralTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

class TensorJoinTest extends NeuralTest {
    @Test void concatenationPreservesValuesAndRepeatedInputGradients() {
        Tensor input = values(2, 3).transpose(0, 1).requiresGrad(true);
        Tensor joined = TensorJoin.concatenate(-1, input, input);
        assertArrayEquals(new int[]{3, 4}, joined.shape());
        assertEquals(input.get(1, 1), joined.get(1, 3));
        joined.sumTensor().backward();
        for (float gradient : input.grad().toFloatArray()) { assertEquals(2, gradient); }
    }

    @Test void stackUsesAnExplicitAxisAndPromotesCompactInputs() {
        Tensor a = values(2, 2).to(DType.FLOAT16).requiresGrad(true);
        Tensor b = values(2, 2).to(DType.BFLOAT16).requiresGrad(true);
        Tensor result = TensorJoin.stack(-1, a, b);
        assertArrayEquals(new int[]{2, 2, 2}, result.shape());
        assertEquals(DType.FLOAT32, result.dtype());
        result.sumTensor().backward();
        assertEquals(DType.FLOAT32, a.grad().dtype());
        assertArrayEquals(new float[]{1, 1, 1, 1}, b.grad().toFloatArray());
    }

    @Test void emptyAxesAndInvalidShapesAreHandled() {
        assertArrayEquals(new int[]{0, 2}, TensorJoin.concatenate(0, new Tensor(0, 2), new Tensor(0, 2)).shape());
        assertThrows(IllegalArgumentException.class, () -> TensorJoin.concatenate(0));
        assertThrows(IllegalArgumentException.class, () -> TensorJoin.stack(0));
        assertThrows(IllegalArgumentException.class, () -> TensorJoin.stack(1, values(2), values(3)));
        assertThrows(IllegalArgumentException.class, () -> TensorJoin.concatenate(1, values(2, 2), values(3, 2)));
        assertThrows(IllegalArgumentException.class, () -> TensorJoin.concatenate(0, values(2, 2), values(2)));
    }
}
