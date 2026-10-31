package io.github.kirstenali.deepj.tensor;

/** Backend-independent metadata transformations; no host download or arithmetic is performed. */
final class TensorViews {
    private TensorViews() {}

    static Tensor unsqueeze(Tensor input, int axis) {
        return Tensor.aliasStorage(input, LayoutTransforms.unsqueeze(input.layout(), axis));
    }

    static Tensor view(Tensor input, int... shape) {
        return Tensor.aliasStorage(input, LayoutTransforms.view(input.layout(), shape));
    }

    static Tensor permute(Tensor input, int... axes) {
        return Tensor.aliasStorage(input, LayoutTransforms.permute(input.layout(), axes));
    }

    static Tensor select(Tensor input, int axis, int index) {
        return Tensor.aliasStorage(input, LayoutTransforms.select(input.layout(), axis, index));
    }

    static Tensor slice(Tensor input, int axis, int start, int stop, int step) {
        return Tensor.aliasStorage(input, LayoutTransforms.slice(input.layout(), axis, start, stop, step));
    }

    static Tensor expand(Tensor input, int... shape) {
        return Tensor.aliasStorage(input, LayoutTransforms.expand(input.layout(), shape));
    }
}
