package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.Autograd;
import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.Arrays;

/** Backend copies plus a slicing autograd rule; never reads tensor values on the host. */
public final class TensorJoin {
    private TensorJoin() {}

    public static Tensor concatenate(int axis, Tensor... inputs) {
        Tensor[] saved = inputs.clone();
        if (saved.length == 0) throw new IllegalArgumentException("Concatenation needs at least one input");
        saved[0].size(axis);
        int dimension = axis < 0 ? saved[0].rank() + axis : axis;
        int[] shape = outputShape(saved, dimension);
        DType dtype = Arrays.stream(saved).map(Tensor::dtype).reduce(DType::promote).orElseThrow();
        try (var scope = TensorScope.open()) {
            Tensor output = Tensor.backend().zeros(dtype, shape);
            copyInputs(output, saved, dimension);
            return scope.escape(Autograd.recordSaved(output, gradient -> split(gradient, saved, dimension), saved));
        }
    }

    public static Tensor stack(int axis, Tensor... inputs) {
        if (inputs.length == 0) throw new IllegalArgumentException("Stack needs at least one input");
        for (Tensor input : inputs) { Tensor.requireSameShape(inputs[0], input, "stack"); }
        Tensor[] expanded = Arrays.stream(inputs).map(input -> input.unsqueeze(axis)).toArray(Tensor[]::new);
        return concatenate(axis, expanded);
    }

    private static int[] outputShape(Tensor[] inputs, int axis) {
        int[] shape = inputs[0].shape();
        shape[axis] = 0;
        for (Tensor input : inputs) {
            requireShape(inputs[0], input, axis);
            shape[axis] = Math.addExact(shape[axis], input.size(axis));
        }
        return shape;
    }

    private static void requireShape(Tensor first, Tensor input, int axis) {
        if (input.rank() != first.rank()) throw new IllegalArgumentException("Concatenation ranks differ");
        for (int dimension = 0; dimension < first.rank(); dimension++) {
            if (dimension != axis && input.size(dimension) != first.size(dimension)) {
                throw new IllegalArgumentException("Concatenation dimensions differ outside the joined axis");
            }
        }
    }

    private static void copyInputs(Tensor output, Tensor[] inputs, int axis) {
        int offset = 0;
        try (var ignored = GradMode.noGrad()) {
            for (Tensor input : inputs) {
                output.slice(axis, offset, offset + input.size(axis)).copyFrom(input.to(output.dtype()));
                offset += input.size(axis);
            }
        }
    }

    private static Tensor[] split(Tensor gradient, Tensor[] inputs, int axis) {
        Tensor[] result = new Tensor[inputs.length];
        int offset = 0;
        for (int index = 0; index < inputs.length; index++) {
            int end = offset + inputs[index].size(axis);
            result[index] = gradient.slice(axis, offset, end);
            offset = end;
        }
        return result;
    }
}
