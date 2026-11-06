package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.autograd.HalfSplitRotary;
import io.github.kirstenali.deepj.autograd.LayoutOps;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Random;
import java.util.stream.IntStream;
import static org.junit.jupiter.api.Assertions.*;

public final class HalfSplitRotaryChecks {
    private HalfSplitRotaryChecks() {}

    public static void compare(DType dtype, boolean strided) {
        Tensor input = Tensor.random(dtype, new Random(73), 12, 128);
        if (strided) input = input.slice(0, 1, 10).slice(1, 0, 128, 2);
        Tensor cosine = Tensor.random(3, input.size(1) / 2, new Random(4));
        Tensor sine = Tensor.random(3, input.size(1) / 2, new Random(5));
        Tensor oldInput = Tensor.from(input.toFloatArray(), dtype, input.shape()).requiresGrad(true);
        input.requiresGrad(true);
        Tensor actual = HalfSplitRotary.apply(input, cosine, sine, 3);
        Tensor expected = reference(oldInput, cosine, sine);
        assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
        actual.multiply(actual.detach()).sumTensor().backward();
        expected.multiply(expected.detach()).sumTensor().backward();
        assertArrayEquals(oldInput.grad().toFloatArray(), input.grad().toFloatArray());
    }

    private static Tensor reference(Tensor input, Tensor cosine, Tensor sine) {
        int width = input.size(1);
        int[] paired = IntStream.range(0, width).map(i -> i / 2 + i % 2 * (width / 2)).toArray();
        int[] original = IntStream.range(0, width).map(i -> i % (width / 2) * 2 + i / (width / 2)).toArray();
        return LayoutOps.rotary(input.indexSelect(-1, paired), cosine, sine, 3, false).indexSelect(-1, original);
    }
}
