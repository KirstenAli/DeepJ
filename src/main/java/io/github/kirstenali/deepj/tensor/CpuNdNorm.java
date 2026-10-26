package io.github.kirstenali.deepj.tensor;

import java.util.List;

/** CPU scalar norm retains the old dense FLOAT32 order and decodes other logical layouts without a shadow. */
final class CpuNdNorm {
    private CpuNdNorm() {}

    static float norm(List<Tensor> tensors) {
        float squares = 0;
        for (Tensor tensor : tensors) { squares = accumulate(tensor, squares); }
        return (float) Math.sqrt(squares);
    }

    private static float accumulate(Tensor input, float squares) {
        input.materialize();
        if (input.dtype() == DType.FLOAT32 && input.coversStorage()) return dense(input.rawData(), squares);
        for (int index = 0; index < input.numel(); index++) {
            float value = input.memory().get(input.layout().addressAt(index));
            squares += value * value;
        }
        return squares;
    }

    private static float dense(float[] values, float squares) {
        for (float value : values) { squares += value * value; }
        return squares;
    }
}
